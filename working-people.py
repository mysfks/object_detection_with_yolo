from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import os
from sort import *

cap = cv2.VideoCapture("Videos/workingPeople.mp4")

model = YOLO('Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "cat", "dog", "cup", "horse", "orange",
              "traffic light", "banana", "telephone", "sofa", "chair", "cake", "mouse", "laptop", "bowl", "bird", "remote",
              "fire hydrant", "stop sign", "parking meter", "bench", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sport ball", "kite", "baseball hat", "fork",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "knife", "spoon", "apple", "sandwich",]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

id_file_path = "person_ids.txt"

# Dosya mevcutsa, mevcut ID'leri oku
if os.path.exists(id_file_path):
    with open(id_file_path, "r") as file:
        existing_ids = set(line.strip() for line in file)
else:
    existing_ids = set()

while True:
    success, img = cap.read()
    if not success:
        break
    
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # KUTULAR İÇİN
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # SINIF İSİMLERİ İÇİN
            cls = int(box.cls[0])
            
            if cls < len(classNames):
                currentClass = classNames[cls]
                
                if currentClass == "person" and conf > 0.3:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)
    
    resultsTracker = tracker.update(detections)
    
    new_ids = set()
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(50, y1)), scale=2, thickness=3, offset=10)
        if str(int(id)) not in existing_ids:
            new_ids.add(int(id))
    
    if new_ids:
        with open(id_file_path, "a") as file:
            for id in new_ids:
                file.write(f"{id}\n")
        existing_ids.update(new_ids)
    
    with open("total_people_count.txt", "w") as file:
        file.write(f"Total number of people detected: {len(existing_ids)}\n")
    
    cv2.imshow('Working People', img)
    cv2.waitKey(1)
