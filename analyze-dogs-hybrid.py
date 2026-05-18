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
DEFAULT_EMBEDDINGS_FILE = "dog_embeddings.npy"
DEFAULT_COLLECTION_ROOT = "tricky-images"
DEFAULT_REVIEW_OUTPUT_DIR = "dog-detection-hybrid"
DEFAULT_DOG_CLASS_ID = 16
BEAR_CLASS_ID = 21  # COCO class ID for 'bear'
DEFAULT_SIMILARITY_THRESHOLD = 0.4  # Adjusted for hybrid space
DEFAULT_YOLO_CONFIDENCE = 0.25

def analyze_collection(
    collection_root: str = DEFAULT_COLLECTION_ROOT,
    embeddings_file: str = DEFAULT_EMBEDDINGS_FILE,
    review_output_dir: str = DEFAULT_REVIEW_OUTPUT_DIR,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    yolo_confidence: float = DEFAULT_YOLO_CONFIDENCE,
    only_save_known: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    collection_root = Path(collection_root)
    review_output_dir = Path(review_output_dir)

    # Load Models
    print(f"Loading YOLO model {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"Loading CLIP model {CLIP_MODEL_NAME}...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

    # 1. Load Image-based Embeddings
    print(f"Loading anchor embeddings from {embeddings_file}...")
    try:
        image_embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading image embeddings: {e}")
        return

    # 2. Generate Text-based Embeddings (CuPL)
    print("Generating text-based embeddings from prompts...")
    dog_names = list(image_embeddings_dict.keys())
    hybrid_embeddings = []

    with torch.no_grad():
        for name in dog_names:
            # Text part
            prompts = DOG_PROMPTS.get(name, [f"A photo of {name}"])
            text_tokens = clip.tokenize(prompts).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            avg_text_feature = text_features.mean(dim=0)
            avg_text_feature /= avg_text_feature.norm(dim=-1, keepdim=True)

            # Image part
            img_feature = torch.from_numpy(image_embeddings_dict[name]).to(device).float()
            
            # Hybrid: Combine (Average) and Re-normalize
            # CLIP space is shared, so we can average image and text vectors
            hybrid_feature = (avg_text_feature + img_feature) / 2.0
            hybrid_feature /= hybrid_feature.norm(dim=-1, keepdim=True)
            hybrid_embeddings.append(hybrid_feature)

    ref_matrix = torch.stack(hybrid_embeddings).to(device).float()

    # Processing images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    if not collection_root.exists():
        print(f"Collection root '{collection_root}' does not exist")
        return
    image_paths = sorted([p for p in collection_root.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])
    
    print(f"Analyzing {len(image_paths)} images (Hybrid Image+Text Approach)...\n")
    print(f"{'Filename':<30} | {'Detected Dogs':<20}")
    print("-" * 55)

    review_output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_paths:
        cv2_img = cv2.imread(str(img_path))
        if cv2_img is None:
            continue

        annotated_img = cv2_img.copy()
        has_dog = False
        has_known_dog = False
        detected_tags = []

        results = yolo_model(cv2_img, conf=yolo_confidence, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if not (cls_id == DEFAULT_DOG_CLASS_ID or (cls_id == BEAR_CLASS_ID and len(result.boxes) == 1)):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_orig, w_orig = cv2_img.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)
                if x2 <= x1 or y2 <= y1: continue

                dog_crop_bgr = cv2_img[y1:y2, x1:x2]
                dog_crop_rgb = cv2.cvtColor(dog_crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(dog_crop_rgb)
                image_input = preprocess(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                similarities = (image_features @ ref_matrix.T).cpu().numpy().flatten()
                all_scores = sorted(zip(dog_names, similarities), key=lambda x: x[1], reverse=True)
                scores_str = ", ".join([f"{name}: {score:.3f}" for name, score in all_scores])

                best_idx = np.argmax(similarities)
                best_score = float(similarities[best_idx])
                best_name = dog_names[best_idx] if best_score >= similarity_threshold else "Unknown Dog"

                detected_tags.append(f"[{scores_str}]")
                has_dog = True
                if best_name != "Unknown Dog": has_known_dog = True

                if (best_name != "Unknown Dog") or (not only_save_known):
                    font_scale = max(0.5, w_orig / 2000.0)
                    thickness = max(1, int(w_orig / 1000.0))
                    label = f"{best_name} | {scores_str}"
                    (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    ty = max(lh + 10, y1 - 10)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                    cv2.rectangle(annotated_img, (x1, ty - lh - bl - 5), (x1 + lw, ty + bl + 5), (0, 255, 0), -1)
                    cv2.putText(annotated_img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        if has_dog and (not only_save_known or has_known_dog):
            cv2.imwrite(str(review_output_dir / img_path.name), annotated_img)

        print(f"{img_path.name:<30} | {', '.join(detected_tags) if detected_tags else 'None'}")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze collection for dogs with Hybrid (Anchor Images + CuPL Text) CLIP + YOLO.")
    parser.add_argument("--collection-root", type=str, default=DEFAULT_COLLECTION_ROOT)
    parser.add_argument("--embeddings-file", type=str, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_REVIEW_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--yolo-confidence", type=float, default=DEFAULT_YOLO_CONFIDENCE)
    parser.add_argument("--only-save-known", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    analyze_collection(
        collection_root=args.collection_root,
        embeddings_file=args.embeddings_file,
        review_output_dir=args.output_dir,
        similarity_threshold=args.threshold,
        yolo_confidence=args.yolo_confidence,
        only_save_known=args.only_save_known,
    )
