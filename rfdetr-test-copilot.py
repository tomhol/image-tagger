#!/usr/bin/env python3

"""
Simple script to test RF-DETR on a directory of images.
Saves results with boxes+labels in an output directory and optional per-image text records.

Usage: python rfdetr-test.py <input-dir> [--model nano|small|medium|large] [--threshold 0.3] [--weights PATH]
"""

import argparse
import sys
from pathlib import Path
import time

import cv2
import numpy as np
import torch

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
    from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES
except ImportError as exc:
    raise ImportError("rfdetr is required for this script. Install it with 'pip install rfdetr'.") from exc

# COCO_CLASS_NAMES is 80-item list (0-based index), COCO_CLASSES is ID->name mapping.
COCO_ID_TO_INDEX = {coco_id: idx for idx, (coco_id, _) in enumerate(sorted(COCO_CLASSES.items()))}

MODEL_MAP = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def normalize_class_ids(class_ids, class_names):
    if class_ids is None:
        return np.array([], dtype=int)
    class_ids = np.asarray(class_ids, dtype=int)
    if class_ids.size == 0:
        return class_ids

    # If IDs are COCO category IDs (1..90, sparse), map them to [0..79] index.
    if np.all(np.isin(class_ids, list(COCO_ID_TO_INDEX.keys()))):
        return np.array([COCO_ID_TO_INDEX[int(x)] for x in class_ids], dtype=int)

    # If IDs are 1-based dense (1..80) we convert to 0-based.
    if class_ids.min() >= 1 and class_ids.max() <= len(class_names):
        return np.clip(class_ids - 1, 0, len(class_names) - 1)

    # If IDs are already 0-based indices, keep them.
    if class_ids.min() >= 0 and class_ids.max() < len(class_names):
        return class_ids

    # Otherwise fallback to clipping within range.
    class_ids = np.clip(class_ids, 0, len(class_names) - 1)
    return class_ids


def draw_detections(image, detections, class_names):
    confidences = detections.confidence if detections.confidence is not None else []
    class_ids = normalize_class_ids(detections.class_id, class_names)

    # Scale font based on image height for hi-res images
    image_height = image.shape[0]
    base_font_scale = 0.5
    font_scale = base_font_scale * max(1.0, image_height / 720)  # Scale up for images taller than 720p

    for xyxy, conf, cls_id in zip(detections.xyxy, confidences, class_ids):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        label = class_names[int(cls_id)] if class_names is not None and len(class_names) > int(cls_id) else str(int(cls_id))
        text = f"{label}: {float(conf):.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get text size with current font scale
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Position text inside bbox at top-left
        text_x = x1 + 2
        text_y = y1 + text_h + 2

        # Draw background rectangle for text (inside bbox)
        cv2.rectangle(image, (x1, y1), (x1 + text_w + 4, y1 + text_h + 4), (0, 255, 0), -1)

        # Draw text
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    return image


def process_directory(src_dir: Path, model_variant: str, threshold: float, weights_path: str | None):
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found or not a directory: {src_dir}")

    dst_dir = src_dir.parent / f"{src_dir.name}-rfdetr-output"
    dst_dir.mkdir(parents=True, exist_ok=True)

    ModelClass = MODEL_MAP.get(model_variant)
    if ModelClass is None:
        raise ValueError(f"Unknown model variant: {model_variant}. Choose one of: {', '.join(MODEL_MAP.keys())}")

    model_kwargs = {"device": "cpu"}
    if weights_path:
        model_kwargs["pretrain_weights"] = str(weights_path)

    print(f"Loading RF-DETR model {model_variant} (device=cpu)...")
    model = ModelClass(**model_kwargs)

    try:
        model.optimize_for_inference(compile=False, batch_size=1, dtype=torch.float32)
        print("Model optimized for inference.")
    except Exception as exc:
        print(f"WARNING: Could not optimize model for inference: {exc}")

    print(f"Model class names count: {len(model.class_names)}")

    image_paths = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif')])
    if not image_paths:
        print("No images found in", src_dir)
        return 0

    for image_path in image_paths:
        print(f"Processing {image_path.name} ...")
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  Could not read {image_path}, skipping.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            start_time = time.perf_counter()
            detections = model.predict(img_rgb, threshold=threshold, verbose=True)
            end_time = time.perf_counter()

            # 4. Calculate and print results
            inference_time = (end_time - start_time) * 1000 # Convert to milliseconds
            print(f"Prediction Timing: {inference_time:.2f} ms")

            # Save annotated image
            out_image = draw_detections(img.copy(), detections, model.class_names)
            out_path = dst_dir / image_path.name
            cv2.imwrite(str(out_path), out_image)

            print(f"  Saved annotated image to {out_path}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  ERROR processing {image_path}: {exc}")
            continue
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run RF-DETR on a folder of images")
    parser.add_argument("directory", help="Source image directory")
    parser.add_argument("--model", default="small", choices=list(MODEL_MAP.keys()), help="RF-DETR variant (default: small)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--weights", default=None, help="Optional local path to RF-DETR weights (default: use pretrained auto-download)")

    args = parser.parse_args()

    try:
        return process_directory(Path(args.directory), args.model, args.threshold, args.weights)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
