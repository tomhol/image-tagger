import os
import cv2
import torch
import clip
import numpy as np
import argparse
import logging
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from iptcinfo3 import IPTCInfo

# Configuration
YOLO_MODEL_PATH = "yolo26l.pt"
CLIP_MODEL_NAME = "ViT-B/32"
EMBEDDINGS_FILE = "dog_embeddings.npy"
DOG_CLASS_ID = 16  # COCO class ID for 'dog' (in fact it is 17 but YOLO is using 0-based indexing, so we use 16 here)
BEAR_CLASS_ID = 21  # COCO class ID for 'bear' (in fact it is 22 but YOLO is using 0-based indexing, so we use 21 here)
MIN_YOLO_CONFIDENCE = 0.4  # default Yolo is 0.25, but we want to be more strict to avoid false positives
MIN_CLIP_SIMILARITY = 0.6
FINAL_TAGS = {
    "saga": "Saga",
    "raff": "Raff",
}
YOLO_CLASS_NAMES = {
    DOG_CLASS_ID: 'dog',
    BEAR_CLASS_ID: 'bear',
}

# Silence iptcinfo3 logger
iptcinfo_logger = logging.getLogger('iptcinfo')
iptcinfo_logger.setLevel(logging.ERROR)

def get_image_paths(directory, recursive=False):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    pattern = "**/*" if recursive else "*"
    paths = []
    for p in Path(directory).glob(pattern):
        if p.is_file() and p.suffix.lower() in valid_extensions:
            paths.append(p)
    return sorted(paths)

def tag_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load YOLO model
    print(f"Loading YOLO model {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # Load CLIP model
    print(f"Loading CLIP model {CLIP_MODEL_NAME}...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    
    # Load reference embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: {EMBEDDINGS_FILE} not found. Please run prepare-clip-embeddings-gemini.py first.")
        return
        
    try:
        ref_embeddings_dict = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return
        
    dog_names = list(ref_embeddings_dict.keys())
    ref_matrix = np.vstack([ref_embeddings_dict[name] for name in dog_names])
    ref_matrix = torch.from_numpy(ref_matrix).to(device).float()
    
    # Get image paths
    image_paths = get_image_paths(args.dir, args.recursive)
    print(f"Found {len(image_paths)} images in {args.dir}")
    
    for img_path in image_paths:
        # Read image
        cv2_img = cv2.imread(str(img_path))
        if cv2_img is None:
            continue

        print(f"{40*'\u2500'}\n{img_path.name} ", end="")
            
        # Run YOLO inference
        results = yolo_model(cv2_img, verbose=args.verbose, conf=args.thresh_yolo)
        
        detected_tags = set()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                # accept either dog, or bear if it's the only detection (to catch Saga/Raff photos where they might be misclassified as bears)
                if cls_id == DOG_CLASS_ID or (cls_id == BEAR_CLASS_ID and len(result.boxes) == 1):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h_orig, w_orig = cv2_img.shape[:2]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    dog_crop_bgr = cv2_img[y1:y2, x1:x2]
                    dog_crop_rgb = cv2.cvtColor(dog_crop_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(dog_crop_rgb)
                    
                    image_input = preprocess(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    similarities = (image_features @ ref_matrix.T).cpu().numpy().flatten()
                    best_idx = np.argmax(similarities)
                    
                    if args.verbose:
                        # print the similarity score for debugging
                        x = [f"{dog_names[i]}: {similarities[i]:.4f}" for i in range(len(dog_names))]
                        print(f"{YOLO_CLASS_NAMES.get(cls_id, f'Unknown ({cls_id})')} scores: {', '.join(x)}")

                    if similarities[best_idx] >= args.thresh_clip:
                        tag = dog_names[best_idx]
                        detected_tags.add(tag)
        
        if detected_tags:
            tags_to_add = list(map(lambda x: FINAL_TAGS.get(x, x), sorted(list(detected_tags))))
            print(f"⭐ {', '.join(tags_to_add)}", end="")
            
            if args.dry_run:
                print("")
            else:
                try:
                    # Save original timestamps
                    stat = os.stat(img_path)
                    atime, mtime = stat.st_atime, stat.st_mtime
                    
                    # Read IPTC info
                    info = IPTCInfo(str(img_path))
                    
                    # Convert byte-strings to regular strings if necessary
                    existing_keywords = [k.decode('utf-8') if isinstance(k, bytes) else k for k in info['keywords']]
                    
                    # Add new tags if not already present
                    changed = False
                    for tag in tags_to_add:
                        if tag not in existing_keywords:
                            info['keywords'].append(tag.encode('utf-8'))
                            changed = True
                    
                    if changed:
                        # Save changes
                        info.save()
                        # iptcinfo3 creates a backup file with ~ suffix, remove it
                        backup_file = str(img_path) + "~"
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                        
                        # Restore timestamps
                        os.utime(img_path, (atime, mtime))
                        print(f"  -> Tags added and timestamps preserved.")
                    else:
                        print(f"  -> Tags already present.")
                        
                except Exception as e:
                    print(f"  -> Error writing metadata for {img_path}: {e}")
        else:
            print("No dogs identified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-tag photos with Saga/Raff using YOLO and CLIP.")
    parser.add_argument("--dir", required=True, help="Directory containing photos to analyze.")
    parser.add_argument("--thresh-clip", type=float, default=MIN_CLIP_SIMILARITY, help=f"CLIP similarity threshold (default: {MIN_CLIP_SIMILARITY}).")
    parser.add_argument("--thresh-yolo", type=float, default=MIN_YOLO_CONFIDENCE, help=f"YOLO confidence threshold (default: {MIN_YOLO_CONFIDENCE}).")
    parser.add_argument("--recursive", action="store_true", help="Search for images recursively.")
    parser.add_argument("--dry-run", action="store_true", help="Analyze photos without modifying metadata.")
    parser.add_argument("--verbose", action="store_true", help="Display detailed information during analysis.")

    args = parser.parse_args()
    tag_images(args)
