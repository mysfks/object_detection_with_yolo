from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("Videos/cars.mp4")

model = YOLO('Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "cat", "dog", "cup", "horse", "orange",
              "traffic light", "banana", "telephone", "sofa", "chair", "cake", "mouse", "laptop", "bowl", "bird", "remote",
              "fire hydrant", "stop sign", "parking meter", "bench", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sport ball", "kite", "baseball hat", "fork",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "knife", "spoon", "apple", "sandwich",] 

mask = cv2.imread('Images/mask.png')

#SAYAÇ
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [200,400,1150,400]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    
    results = model(imgRegion, stream=True)
    
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #KUTULAR İÇİN
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            w, h = x2 - x1, y2 - y1
            
            #DOĞRULUK ORANLARI İÇİN
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            #SINIF İSİMLERİ İÇİN
            cls = int(box.cls[0])
            
            if cls < len(classNames):
                currentClass = classNames[cls]
                
                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                    #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(50, y1)), scale=1, thickness=1, offset=3)
                    #cvzone.cornerRect(img, (x1, y1, w, h), l=8)
                    
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
    
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(50, y1)), scale=2, thickness=3, offset=10)
        
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        
        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), 5)
            
    cvzone.putTextRect(img, f'Count: {int(len(totalCount))}', (50, 50))        
     
    cv2.imshow('Cars Passing', img)
    #cv2.imshow('Cars Region', imgRegion)
    cv2.waitKey(1)