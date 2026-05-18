import cv2
import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# Configuration
YOLO_MODEL_PATH = "yolo26l.pt"
CLIP_MODEL_NAME = "ViT-B/32"
EMBEDDINGS_FILE = "dog_embeddings.npy"
COLLECTION_ROOT = Path("collection")
DOG_CLASS_ID = 16  # COCO class ID for 'dog'
SIMILARITY_THRESHOLD = 0.6  # Adjust based on performance

def analyze_collection():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load YOLO model
    print(f"Loading YOLO model {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # Load CLIP model
    print(f"Loading CLIP model {CLIP_MODEL_NAME}...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    
    # Load reference embeddings
    print(f"Loading reference embeddings from {EMBEDDINGS_FILE}...")
    try:
        # np.load for a dict saved with np.save(file, dict) requires allow_pickle=True
        ref_embeddings_dict = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return
        
    # Prepare reference names and matrices for fast comparison
    dog_names = list(ref_embeddings_dict.keys())
    ref_matrix = np.vstack([ref_embeddings_dict[name] for name in dog_names])
    ref_matrix = torch.from_numpy(ref_matrix).to(device).float()
    
    # Supported extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    image_paths = sorted([p for p in COLLECTION_ROOT.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])
    
    print(f"Analyzing {len(image_paths)} images in {COLLECTION_ROOT}...\n")
    print(f"{'Filename':<30} | {'Detected Dogs':<20}")
    print("-" * 55)
    
    for img_path in image_paths:
        # Read image with OpenCV for YOLO
        cv2_img = cv2.imread(str(img_path))
        if cv2_img is None:
            continue
            
        # Run YOLO inference
        results = yolo_model(cv2_img, verbose=False)
        
        detected_tags = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == DOG_CLASS_ID:
                    # Get bbox coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Ensure coordinates are within image boundaries
                    h_orig, w_orig = cv2_img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_orig, x2), min(h_orig, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    # Crop the dog from the image
                    # Note: OpenCV uses BGR, CLIP/PIL expects RGB
                    dog_crop_bgr = cv2_img[y1:y2, x1:x2]
                    dog_crop_rgb = cv2.cvtColor(dog_crop_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(dog_crop_rgb)
                    
                    # Preprocess for CLIP
                    image_input = preprocess(pil_img).unsqueeze(0).to(device)
                    
                    # Generate CLIP embedding
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    # Compute cosine similarity with reference embeddings
                    # Both are normalized, so we can use dot product
                    similarities = (image_features @ ref_matrix.T).cpu().numpy().flatten()
                    
                    # Find the best match
                    best_idx = np.argmax(similarities)
                    if similarities[best_idx] >= SIMILARITY_THRESHOLD:
                        detected_tags.append(dog_names[best_idx])
                    else:
                        detected_tags.append("Unknown Dog")
        
        # Output the results
        tags_str = ", ".join(sorted(list(set(detected_tags)))) if detected_tags else "None"
        print(f"{img_path.name:<30} | {tags_str}")

if __name__ == "__main__":
    analyze_collection()
