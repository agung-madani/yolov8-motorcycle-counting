# YOLOV8 Motorcycle/Motorbikes Counter

This repository contains a script for tracking motorcycles in a video using YOLO (You Only Look Once) object detection. The script is designed to process a specific video file and perform motorcycle tracking and counting based on predefined areas.

https://github.com/agung-madani/yolov8-motorcycle-counting/assets/121701309/fea95f91-9c25-49e2-a065-ac47c36aa1dd

## Files

- `coco.txt`: Contains class names used for object detection.
- `gerbanguin.mp4`: Sample video file for tracking motorcycles.
- `main.py`: Python script for motorcycle tracking using YOLO and custom tracking logic.
- `README.md`: This file, providing information about the repository.
- `result counter.mp4`: Sample output video showing the counted motorcycles.
- `tracker.py`: Python script containing the custom Tracker class for object tracking.
- `yolov8s.pt`: Pretrained weight file used for the YOLO model.

## Requirements

Make sure you have the following libraries installed:

- Python 3
- OpenCV
- Pandas
- cvzone
- Ultralytics

You can install the required Python packages using pip:

```BASH
pip install opencv-python pandas cvzone ultralytics
```
## Usage

To run the motorcycle tracking script, ensure that you have all the necessary files in the same directory. The script uses OpenCV, Pandas, and cvzone libraries for image processing and tracking. Make sure you have these libraries installed.

You can run the script `main.py` with Python 3. Make sure to provide the necessary video file and weight file as required. The script will display the processed video with motorcycle tracking and counting information.

## Author

- Agung Rashif Madani
