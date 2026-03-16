# image-tagger

My work-in-progress project is intended for experimentation with ML-assisted:
- object detection
- image classification

The goal here is to (eventually) implement a set of Python scripts to process my large photo collection (since ~2002) and add tags (IPTC metadata) based on custom criteria.

For example, over these many years I have had several dogs, and I want to tag all photos where a particular dog is present.

So the processing pipeline will be:
- use YOLO (or similar) to detect dogs in the photo
- use CLIP (or similar) to differentiate individual dogs and generate tags based on the particular dog name
- save the tags to IPTC metadata in the original image