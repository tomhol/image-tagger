#!/usr/bin/env python3

"""
Simple script to test RF-DETR on a directory of photos. 
It saves the results with boxes and labels in a new directory.
Similar to yolo-test.py but using the superior RF-DETR model.
"""

import sys
from pathlib import Path
import cv2
import supervision as sv
from rfdetr import RFDETRNano # you can change to Nano, Medium, Large
from rfdetr.assets.coco_classes import COCO_CLASSES

# Model options: "nano", "small", "medium", "large"
MODEL_NAME = "nano" 

def get_model(name: str):
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
    models = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    return models.get(name.lower(), RFDETRSmall)

def main(directory: str):
    SRC_DIR = Path(directory)

    if not SRC_DIR.exists():
        print(f"Error: source directory not found at {SRC_DIR}!", file=sys.stderr)
        return 1
    if not SRC_DIR.is_dir():
        print(f"Error: the source directory {SRC_DIR} is not a directory!", file=sys.stderr)
        return 1

    DST_DIR = Path(directory + "-rfdetr-output")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading RF-DETR model: {MODEL_NAME} ...")
    ModelClass = get_model(MODEL_NAME)
    model = ModelClass(device="cpu") # RF-DETR is efficient on CPU

    try:
        model.optimize_for_inference(compile=False, batch_size=1)
        print("Model optimized for inference.")
    except Exception as e:
        print(f"Optimization warning: {e}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in image_extensions])

    if not image_paths:
        print(f"No images found in {SRC_DIR}")
        return 0

    print(f"Processing {len(image_paths)} images ...")

    # Map COCO category IDs to labels. COCO IDs are sparse (1-90).
    # model.class_names is a dense list of 80, but model returns sparse IDs.
    label_map = COCO_CLASSES 

    for image_path in image_paths:
        print(f"Predicting: {image_path.name}")
        
        # RF-DETR returns sparse COCO IDs in detections.class_id
        detections = model.predict(str(image_path), threshold=0.3)
        
        # Annotate
        image = cv2.imread(str(image_path))
        if image is None:
            continue
            
        h, w, _ = image.shape
        # Scale font and thickness based on image resolution
        dynamic_scale = max(1.0, w / 1200)
        dynamic_thickness = max(1, int(dynamic_scale * 1.5))

        box_annotator = sv.BoxAnnotator(thickness=dynamic_thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=dynamic_scale, 
            text_thickness=dynamic_thickness,
            text_padding=int(10 * dynamic_scale),
            text_position=sv.Position.TOP_LEFT,
        )

        annotated_image = box_annotator.annotate(
            scene=image.copy(), 
            detections=detections
        )
        
        # Prepare labels using the COCO ID mapping
        labels = [
            f"{label_map.get(int(class_id), f'ID {class_id}')} {confidence:.2f}"
            for class_id, confidence 
            in zip(detections.class_id, detections.confidence)
        ]
        
        annotated_image = label_annotator.annotate(
            scene=annotated_image, 
            detections=detections,
            labels=labels
        )
        
        # Save result
        out_path = DST_DIR / image_path.name
        cv2.imwrite(str(out_path), annotated_image)

    print(f"Done! Results saved to {DST_DIR}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1].rstrip("/"))
