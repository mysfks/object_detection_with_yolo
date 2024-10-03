from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("Videos/people.mp4")

model = YOLO('Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "cat", "dog", "cup", "horse", "orange",
              "traffic light", "banana", "telephone", "sofa", "chair", "cake", "mouse", "laptop", "bowl", "bird", "remote",
              "fire hydrant", "stop sign", "parking meter", "bench", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sport ball", "kite", "baseball hat", "fork",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "knife", "spoon", "apple", "sandwich",] 

mask = cv2.imread('Images/maskPeople.png')

#SAYAÇ
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [200,200,1150,200]
limitsDown = [200,550,1150,550]
totalCountUp = []
totalCountDown = []

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
                
                if currentClass in ["person"] and conf > 0.3:
                    #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(50, y1)), scale=1, thickness=1, offset=3)
                    #cvzone.cornerRect(img, (x1, y1, w, h), l=8)
                    
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
    
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0,0,255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0,0,255), 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(50, y1)), scale=2, thickness=3, offset=10)
        
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-20 < cy < limitsUp[1]+20:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0,255,0), 5)
                
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-20 < cy < limitsDown[1]+20:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0,255,0), 5)
            
    cv2.putText(img, str(len(totalCountUp)), (50,50), cv2.FONT_HERSHEY_PLAIN, 5, (139,195,75), 7)
    cv2.putText(img, str(len(totalCountDown)), (50,110), cv2.FONT_HERSHEY_PLAIN, 5, (50,50,230), 7)
    #cvzone.putTextRect(img, f'Count Up: {int(len(totalCountUp))}', (50, 50))
    #cvzone.putTextRect(img, f'Count Down: {int(len(totalCountDown))}', (50, 100))          
     
    cv2.imshow('Cars Passing', img)
    #cv2.imshow('Cars Region', imgRegion)
    cv2.waitKey(1)