# Import necessary libraries
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker  # Import the Tracker class from tracker.py
import cvzone

# Initialize the YOLO model with a pretrained weight file
model = YOLO('yolov8s.pt')

# Define a function to handle mouse events
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create an OpenCV window named 'RGB' and register the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file for processing
cap = cv2.VideoCapture('gerbanguin.mp4')

# Read class names from a file named "coco.txt"
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Initialize counters and dictionaries to track motorcycle entering and exiting defined areas
count = 0
tracker = Tracker()
# eft-bottom, left-top, right-top, right-bottom

area1 = [(433, 328), (475, 313), (882, 329), (874, 353)] # red
area2 = [(488, 310), (537, 302), (889, 313), (886, 325)] # green
motorcycle_enter = {}
counter1 = []
motorcycle_exit = {}
counter2 = []

# Main loop for processing each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection using the YOLO model
    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'motorcycle' in c:
            list.append([x1, y1, x2, y2])
    # Update the tracker with the list of detected motorcycle
    bbox_id = tracker.update(list)

    # Draw bounding boxes and IDs on tracked objects
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox

        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 1)
        cv2.circle(frame, ((x3+((x4-x3)//2)), y4), 2, (255, 0, 0), -1)
        cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)

        # motorcycle Exiting
        results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        if results >= 0:
            motorcycle_exit[id] = (x4, y4)
        if id in motorcycle_exit:
            results1 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 1)
                cv2.circle(frame, (x3+((x4-x3)//2), y4), 2, (255, 0, 0), -1)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter2.count(id) == 0:
                    counter2.append(id)

        # motorcycle entering
        results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results2 >= 0:
            motorcycle_enter[id] = (x4, y4)
        if id in motorcycle_enter:
            results3 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 1)
                cv2.circle(frame, (x3+((x4-x3)//2), y4), 2, (255, 0, 0), -1)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if counter1.count(id) == 0:
                    counter1.append(id)

    # Draw polylines to define areas
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 1)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 1)
    entr = len(counter1)
    ext = len(counter2)
    cvzone.putTextRect(frame, f'Motorbikes enters the UIN Gate: {entr}', (50, 90), 1, 1, (255, 255, 255), (255, 165, 0))
    cvzone.putTextRect(frame, f'Motorbikes exits the UIN Gate: {ext}', (50, 130), 1, 1, (255, 255, 255), (255, 165, 0))
    cvzone.putTextRect(frame, f'Programmer: Agung Rashif Madani', (688, 465), 1, 1, (255, 255, 255), (128, 0, 128))
    cvzone.putTextRect(frame, f'Motorbikes Counter', (370, 50), 2, 2, (255, 255, 255), (2, 243, 152))

    # Display the processed frame in the 'RGB' window
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
