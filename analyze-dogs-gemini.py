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

# Configuration
YOLO_MODEL_PATH = "yolo26l.pt"
CLIP_MODEL_NAME = "ViT-B/32"
EMBEDDINGS_FILE = "cakun_embeddings.npy"
DOG_CLASS_ID = 16  # COCO class ID for 'dog'
BEAR_CLASS_ID = 21  # COCO class ID for 'bear'
MIN_YOLO_CONFIDENCE = 0.4
DEFAULT_OUTPUT_DIR = "dog-detection-gemini"

def get_image_paths(directory, recursive=False):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    pattern = "**/*" if recursive else "*"
    paths = []
    for p in Path(directory).glob(pattern):
        if p.is_file() and p.suffix.lower() in valid_extensions:
            paths.append(p)
    return sorted(paths)

def analyze_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving analyzed images to: {output_dir}")
    
    # Load YOLO model
    print(f"Loading YOLO model {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # Load CLIP model
    print(f"Loading CLIP model {CLIP_MODEL_NAME}...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    
    # Load reference embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Error: {EMBEDDINGS_FILE} not found. Please ensure the embeddings file exists.")
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

        # Run YOLO inference
        results = yolo_model(cv2_img, verbose=False, conf=args.thresh_yolo)
        
        has_detection = False
        annotated_img = cv2_img.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                # accept either dog, or bear if it's the only detection
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
                    best_score = similarities[best_idx]
                    best_label = dog_names[best_idx]
                    
                    # Draw rectangle and label
                    color = (0, 255, 0) # Green
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
                    
                    label_text = f"{best_label}: {best_score:.3f}"
                    # Calculate text size to put a background box for readability
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    cv2.putText(annotated_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    has_detection = True
                    print(f"Detected {best_label} ({best_score:.3f}) in {img_path.name}")
        
        if has_detection:
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), annotated_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze photos for dogs and generate labeled output images.")
    parser.add_argument("--dir", required=True, help="Directory containing photos to analyze.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory to save analyzed images (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--thresh-yolo", type=float, default=MIN_YOLO_CONFIDENCE, help=f"YOLO confidence threshold (default: {MIN_YOLO_CONFIDENCE}).")
    parser.add_argument("--recursive", action="store_true", help="Search for images recursively.")

    args = parser.parse_args()
    analyze_images(args)