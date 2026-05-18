import argparse
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from dog_prompts import DOG_PROMPTS

# Configuration
YOLO_MODEL_PATH = "yolo26l.pt"
CLIP_MODEL_NAME = "ViT-B/32"
DEFAULT_COLLECTION_ROOT = "tricky-images"
DEFAULT_REVIEW_OUTPUT_DIR = "dog-detection-cupl"
DEFAULT_DOG_CLASS_ID = 16  # COCO class ID for 'dog'
BEAR_CLASS_ID = 21  # COCO class ID for 'bear'
DEFAULT_SIMILARITY_THRESHOLD = 0.25  # Text-image similarity is typically lower than image-image
DEFAULT_YOLO_CONFIDENCE = 0.25

def analyze_collection(
    collection_root: str = DEFAULT_COLLECTION_ROOT,
    review_output_dir: str = DEFAULT_REVIEW_OUTPUT_DIR,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    yolo_confidence: float = DEFAULT_YOLO_CONFIDENCE,
    only_save_known: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    collection_root = Path(collection_root)
    review_output_dir = Path(review_output_dir)

    # Load YOLO model
    print(f"Loading YOLO model {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # Load CLIP model
    print(f"Loading CLIP model {CLIP_MODEL_NAME}...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

    # Generate reference text embeddings from CuPL prompts
    print("Generating reference text embeddings from descriptive prompts...")
    dog_names = list(DOG_PROMPTS.keys())
    ref_embeddings = []
    
    with torch.no_grad():
        for name in dog_names:
            prompts = DOG_PROMPTS[name]
            text_tokens = clip.tokenize(prompts).to(device)
            text_features = clip_model.encode_text(text_tokens)
            # Normalize and average the prompt embeddings for this class
            text_features /= text_features.norm(dim=-1, keepdim=True)
            avg_feature = text_features.mean(dim=0)
            avg_feature /= avg_feature.norm(dim=-1, keepdim=True)
            ref_embeddings.append(avg_feature)
            
    # Stack into a reference matrix
    ref_matrix = torch.stack(ref_embeddings).to(device).float()

    # Supported extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    if not collection_root.exists():
        print(f"Collection root '{collection_root}' does not exist")
        return
    image_paths = sorted([p for p in collection_root.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])
    
    print(f"Analyzing {len(image_paths)} images in {collection_root} (CuPL Approach)...\n")
    print(f"{'Filename':<30} | {'Detected Dogs':<20}")
    print("-" * 55)

    review_output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_paths:
        # Read image with OpenCV for YOLO
        cv2_img = cv2.imread(str(img_path))
        if cv2_img is None:
            continue

        annotated_img = cv2_img.copy()
        has_dog = False
        has_known_dog = False
        detected_tags = []

        # Run YOLO inference
        results = yolo_model(cv2_img, conf=yolo_confidence, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if not (cls_id == DEFAULT_DOG_CLASS_ID or (cls_id == BEAR_CLASS_ID and len(result.boxes) == 1)):
                    continue

                # Get bbox coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Ensure coordinates are within image boundaries
                h_orig, w_orig = cv2_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_orig, x2), min(h_orig, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop the dog from the image
                dog_crop_bgr = cv2_img[y1:y2, x1:x2]
                dog_crop_rgb = cv2.cvtColor(dog_crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(dog_crop_rgb)

                # Preprocess for CLIP
                image_input = preprocess(pil_img).unsqueeze(0).to(device)

                # Generate CLIP embedding for the image crop
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                # Compute cosine similarity with the averaged text reference embeddings
                similarities = (image_features @ ref_matrix.T).cpu().numpy().flatten()
                
                # Create a list of all scores for debugging
                all_scores = sorted(zip(dog_names, similarities), key=lambda x: x[1], reverse=True)
                scores_str = ", ".join([f"{name}: {score:.3f}" for name, score in all_scores])

                best_idx = np.argmax(similarities)
                best_score = float(similarities[best_idx])
                best_name = dog_names[best_idx] if best_score >= similarity_threshold else "Unknown Dog"

                # Update detected_tags with all scores for this detection
                detection_label = f"[{scores_str}]"
                detected_tags.append(detection_label)

                # Mark and annotate image
                has_dog = True
                if best_name != "Unknown Dog":
                    has_known_dog = True

                draw_annotation = (best_name != "Unknown Dog") or (not only_save_known)
                if draw_annotation:
                    # Calculate dynamic scaling based on image width
                    font_scale = max(0.5, w_orig / 2000.0)
                    thickness = max(1, int(w_orig / 1000.0))

                    # For the image label, we'll show the best match plus the full scores list
                    label = f"{best_name} | {scores_str}"
                    
                    # Get text size for background rectangle
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # Ensure label is within image bounds
                    text_y = max(label_height + 10, y1 - 10)
                    back_y1 = text_y - label_height - baseline - 5
                    back_y2 = text_y + baseline + 5
                    
                    # Draw bounding box, background rectangle, and text
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                    cv2.rectangle(annotated_img, (x1, back_y1), (x1 + label_width, back_y2), (0, 255, 0), -1)
                    cv2.putText(
                        annotated_img,
                        label,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA,
                    )

        should_save_image = has_dog and (not only_save_known or has_known_dog)
        if should_save_image:
            output_path = review_output_dir / img_path.name
            cv2.imwrite(str(output_path), annotated_img)

        tags_str = ", ".join(detected_tags) if detected_tags else "None"
        print(f"{img_path.name:<30} | {tags_str}")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze collection for dogs with CuPL (Text-based CLIP) + YOLO.")
    parser.add_argument("--collection-root", type=str, default=DEFAULT_COLLECTION_ROOT, help="Path to image collection")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_REVIEW_OUTPUT_DIR, help="Directory to save annotated images")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Cosine similarity threshold for assignment")
    parser.add_argument("--yolo-confidence", type=float, default=DEFAULT_YOLO_CONFIDENCE, help="YOLO detection confidence threshold")
    parser.add_argument("--only-save-known", action="store_true", help="Only save images when known dog match above threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_collection(
        collection_root=args.collection_root,
        review_output_dir=args.output_dir,
        similarity_threshold=args.threshold,
        yolo_confidence=args.yolo_confidence,
        only_save_known=args.only_save_known,
    )
