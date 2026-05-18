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
EMBEDDINGS_FILE = "cakun_embeddings.npy"
COLLECTION_ROOT = Path("tricky-images")
OUTPUT_DIR = Path("dog-detection-gemini-cli")  # Updated output directory
DOG_CLASS_ID = 16  # COCO class ID for 'dog'
SIMILARITY_THRESHOLD = 0.6  # Adjust based on performance

def analyze_collection():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
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
    
    for img_path in image_paths:
        # Read image with OpenCV for YOLO and drawing
        cv2_img = cv2.imread(str(img_path))
        if cv2_img is None:
            continue
            
        # Run YOLO inference
        results = yolo_model(cv2_img, verbose=False)
        
        dog_found = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == DOG_CLASS_ID:
                    dog_found = True
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
                    similarities = (image_features @ ref_matrix.T).cpu().numpy().flatten()
                    
                    # Find the best match
                    best_idx = np.argmax(similarities)
                    sim_score = similarities[best_idx]
                    
                    if sim_score >= SIMILARITY_THRESHOLD:
                        label = f"{dog_names[best_idx]} ({sim_score:.2f})"
                        color = (0, 255, 0) # Green for match
                    else:
                        label = f"Unknown Dog ({sim_score:.2f})"
                        color = (0, 0, 255) # Red for unknown
                    
                    # Draw rectangle and label
                    cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(cv2_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Save the image if at least one dog was found
        if dog_found:
            output_path = OUTPUT_DIR / img_path.name
            cv2.imwrite(str(output_path), cv2_img)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    analyze_collection()
