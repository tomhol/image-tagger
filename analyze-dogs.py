import os
# Configure Keras to use PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import cv2
import numpy as np
import keras
import torch
import logging
import fnmatch
from pathlib import Path
from ultralytics import YOLO
from rfdetr import RFDETRNano
from iptcinfo3 import IPTCInfo

# Configuration matching training script
IMG_SIZE = (224, 224)
MODEL_PATH = "dog_classifier.keras"
CLASSES_PATH = "dog_classes.txt"

# Default constants
YOLO_MODEL_PATH = "yolo26l.pt"
DEFAULT_COLLECTION_ROOT = "tricky-images"
DEFAULT_REVIEW_OUTPUT_DIR = "dog-detection-keras"
DOG_ID_YOLO = 16  # Yolo uses 80-based COCO class IDs, where dog is 17th class (index 16 in 0-based)
DOG_ID_DETR = 18  # DETR uses standard COCO class IDs, dog is traditionally 18
DEFAULT_CONFIDENCE = 0.25
DEFAULT_CLASS_THRESHOLD = 0.5
MAX_PREVIEW_SIZE = 1600

# Silence iptcinfo3 logger
iptcinfo_logger = logging.getLogger('iptcinfo')
iptcinfo_logger.setLevel(logging.ERROR)

def load_class_names(path):
    if not Path(path).exists():
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def analyze_collection(
    collection_root: str,
    output_dir: str,
    detector_type: str,
    confidence: float,
    threshold: float,
    mode: str,
    filter_pattern: str = None,
    valid_tags: str = None,
):
    collection_root = Path(collection_root)
    output_dir = Path(output_dir)

    # Check for trained model
    if not Path(MODEL_PATH).exists() or not Path(CLASSES_PATH).exists():
        print(f"Error: Model '{MODEL_PATH}' or classes file '{CLASSES_PATH}' not found.")
        print("Please run 'train_dog_classifier.py' first.")
        return

    # Initialize Detector
    detector = None
    if detector_type == "yolo":
        print(f"Loading YOLO model {YOLO_MODEL_PATH}...")
        detector = YOLO(YOLO_MODEL_PATH)
    elif detector_type == "detr":
        print(f"Loading RF-DETR model \"nano\"...")
        detector = RFDETRNano(device="cpu")
        detector.optimize_for_inference(compile=False, batch_size=1, dtype=torch.float32)
    else:
        print(f"Error: Unknown detector type '{detector_type}'")
        return

    print(f"Loading Keras model {MODEL_PATH}...")
    keras_model = keras.models.load_model(MODEL_PATH)
    class_names = load_class_names(CLASSES_PATH)
    print(f"Known classes: {class_names}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    if not collection_root.exists():
        print(f"Collection root '{collection_root}' does not exist")
        return
        
    if filter_pattern:
        image_paths = sorted([p for p in collection_root.glob(filter_pattern) if p.is_file() and p.suffix.lower() in valid_extensions])
    else:
        image_paths = sorted([p for p in collection_root.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])

    print(f"Analyzing {len(image_paths)} images using {detector_type.upper()} (Mode: {mode})...\n")

    if mode == "analyze":
        output_dir.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    stats = {
        "processed": 0,
        "with_dogs": 0,
        "detections": 0,
        "sum_conf": 0.0,
        "tagged": 0
    }

    # Process valid tags
    valid_tags_list = None
    if valid_tags:
        valid_tags_list = [t.strip().lower() for t in valid_tags.split(",")]

    for img_path in image_paths:
        cv2_img = cv2.imread(str(img_path))
        if cv2_img is None:
            continue

        stats["processed"] += 1
        annotated_img = cv2_img.copy() if mode == "analyze" else None
        has_dog = False
        tags_to_apply = set()
        print_labels = []

        # Unified detection handling
        detections = []
        if detector_type == "yolo":
            results = detector(cv2_img, conf=confidence, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == DOG_ID_YOLO:
                        detections.append({
                            "bbox": map(int, box.xyxy[0]),
                            "conf": float(box.conf[0])
                        })
        else: # detr
            img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            results = detector.predict(img_rgb, threshold=confidence, verbose=False)
            if results.xyxy is not None:
                for i in range(len(results.xyxy)):
                    cls_id = int(results.class_id[i])
                    if cls_id == DOG_ID_DETR:
                        detections.append({
                            "bbox": map(int, results.xyxy[i]),
                            "conf": float(results.confidence[i])
                        })

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            h_orig, w_orig = cv2_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop and preprocess for Keras
            crop_bgr = cv2_img[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(crop_rgb, IMG_SIZE)
            input_array = np.expand_dims(resized_img, axis=0)

            # Predict class
            preds = keras_model.predict(input_array, verbose=0)
            score = np.max(preds[0])
            class_idx = np.argmax(preds[0])
            
            label = class_names[class_idx]
            if score < threshold:
                label = "Unknown"

            # Filter by valid_tags
            if valid_tags_list and label.lower() not in valid_tags_list:
                continue

            stats["detections"] += 1
            stats["sum_conf"] += score
            has_dog = True
            
            label_text = f"{label} ({score:.2f})"
            print_labels.append(label_text)
            tags_to_apply.add(label)

            # Annotation
            if mode == "analyze":
                color = (0, 255, 0)
                text_color = (0, 0, 0)
                font_scale = max(0.5, w_orig / 2000.0)
                thickness = max(1, int(w_orig / 1000.0))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(annotated_img, (x1, y1), (x1 + text_w + 10, y1 + text_h + baseline + 10), color, -1)
                cv2.putText(annotated_img, label_text, (x1 + 5, y1 + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        if has_dog:
            stats["with_dogs"] += 1
            print(f"{img_path.name:<30} | {', '.join(print_labels)}")
            
            if mode == "analyze":
                h, w = annotated_img.shape[:2]
                if max(h, w) > MAX_PREVIEW_SIZE:
                    scale = MAX_PREVIEW_SIZE / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    annotated_img = cv2.resize(annotated_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(output_dir / img_path.name), annotated_img)
            
            elif mode == "tag-images" and tags_to_apply:
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
                    for tag in sorted(list(tags_to_apply)):
                        if tag not in existing_keywords:
                            info['keywords'].append(tag.encode('utf-8'))
                            changed = True
                    
                    if changed:
                        info.save()
                        # Remove backup file
                        backup_file = str(img_path) + "~"
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                        
                        # Restore timestamps
                        os.utime(img_path, (atime, mtime))
                        stats["tagged"] += 1
                        print(f"  -> Tags added: {', '.join(tags_to_apply)}")
                    else:
                        print(f"  -> Tags already present.")
                        
                except Exception as e:
                    print(f"  -> Error writing metadata for {img_path}: {e}")

    # Print summary statistics
    print(f"\n{40*'\u2500'}")
    print(f"Summary Statistics:")
    print(f"  Images processed:      {stats['processed']}")
    print(f"  Images with dogs:     {stats['with_dogs']}")
    print(f"  Total detections:     {stats['detections']}")
    if stats["detections"] > 0:
        print(f"  Average confidence:   {stats['sum_conf'] / stats['detections']:.4f}")
    if mode == "tag-images":
        print(f"  Images tagged:        {stats['tagged']}")
    print(f"{40*'\u2500'}")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dogs with YOLO/DETR detection and Keras classification.")
    parser.add_argument("--collection-root", type=str, default=DEFAULT_COLLECTION_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_REVIEW_OUTPUT_DIR)
    parser.add_argument("--detector", type=str, choices=["yolo", "detr"], default="detr", help="Detection model to use (default: detr)")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="Detection confidence threshold")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CLASS_THRESHOLD, help="Classification confidence threshold")
    parser.add_argument("--mode", type=str, choices=["analyze", "dry-run", "tag-images"], default="analyze", help="Operational mode")
    parser.add_argument("--filter", type=str, default=None, help="Glob pattern to filter images (e.g. '2023-12-*.jpg')")
    parser.add_argument("--valid-tags", type=str, default=None, help="Comma-separated list of valid tags (e.g. 'Saga,Raff')")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    analyze_collection(
        collection_root=args.collection_root,
        output_dir=args.output_dir,
        detector_type=args.detector,
        confidence=args.confidence,
        threshold=args.threshold,
        mode=args.mode,
        filter_pattern=args.filter,
        valid_tags=args.valid_tags,
    )
