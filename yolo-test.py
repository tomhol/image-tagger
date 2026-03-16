#!/usr/bin/env python3

"""
Simple script to test YOLO on a directory of photos. 
It saves the results with boxes and labels in a new directory.
"""

import sys
from pathlib import Path
from ultralytics import YOLO


# MODEL_NAME = "yolov8l.pt"
# MODEL_NAME = "yolo11l.pt"
MODEL_NAME = "yolo26l.pt"


def main(directory: str):
    SRC_DIR = Path(directory)

    if not SRC_DIR.exists():
        print(f"Error: source directory not found at {SRC_DIR}!", file=sys.stderr)
        return 1
    if not SRC_DIR.is_dir():
        print(f"Error: the source directory {SRC_DIR} is not a directory!", file=sys.stderr)
        return 1

    DST_DIR = Path(directory + "-yolo-output")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model: {MODEL_NAME} ...")
    model = YOLO(MODEL_NAME)

    # note: using `batch` is tricky, because it converts all images to the same **square** size 
    # (by default 640x640) and then pictures are then typically distorted (squashed) and the inference
    # detects slightly suboptimal results sometimes.
    results_generator = model.predict(
        f"{SRC_DIR}/*.[jJ][pP][gG]", 
        verbose=True, 
        stream=True,
        # batch=4,  # see the note above
    )

    for result in results_generator:
        # saves with boxes and labels by default
        result.save(str(DST_DIR / Path(result.path).name)) 

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1].rstrip("/"))
