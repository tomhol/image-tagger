import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# Configuration
MODEL_PATH = "yolo26l.pt"  # Use the user-specified model
INPUT_ROOT = Path("/mnt/c/temp/Offload")
OUTPUT_ROOT = Path("training-set")
TARGET_HEIGHT = 720
DOG_CLASS_ID = 16  # COCO class ID for 'dog'

def process_images():
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Ensure output root exists
    OUTPUT_ROOT.mkdir(exist_ok=True)
    
    # Iterate through sub-folders (raff, saga, ...)
    for dog_name in ["Raff", "saga", "Cakun"]:
        input_dir = INPUT_ROOT / dog_name
        output_dir = OUTPUT_ROOT / dog_name
        
        if not input_dir.exists():
            print(f"Skipping {dog_name}: directory not found.")
            continue
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported extensions
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
        image_paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]
        
        print(f"Processing {len(image_paths)} images for {dog_name}...")
        
        for img_path in image_paths:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read {img_path}")
                continue
                
            # Run inference
            results = model(img, verbose=False)
            
            # Find the largest dog bounding box
            max_area = 0
            best_bbox = None
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Check if class is dog (COCO ID 16)
                    cls_id = int(box.cls[0])
                    if cls_id == DOG_CLASS_ID:
                        # Get bbox coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        if area > max_area:
                            max_area = area
                            best_bbox = (x1, y1, x2, y2)
            
            if best_bbox:
                x1, y1, x2, y2 = best_bbox
                
                # Ensure coordinates are within image boundaries
                h_orig, w_orig = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_orig, x2), min(h_orig, y2)
                
                # Crop
                cropped_img = img[y1:y2, x1:x2]
                
                if cropped_img.size == 0:
                    print(f"Invalid crop for {img_path}")
                    continue
                
                # Resize to target height maintaining aspect ratio
                h_crop, w_crop = cropped_img.shape[:2]
                aspect_ratio = w_crop / h_crop
                new_width = int(TARGET_HEIGHT * aspect_ratio)
                
                resized_img = cv2.resize(cropped_img, (new_width, TARGET_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                
                # Save
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), resized_img)
                print(f"Saved: {output_path}")
            else:
                print(f"No dog detected in: {img_path}")

if __name__ == "__main__":
    process_images()
