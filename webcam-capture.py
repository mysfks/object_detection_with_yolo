from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(0)  WEB CAM İÇİN KULLANILMASI GEREKEN KODLAR
#cap.set(3, 640)
#cap.set(4, 480)

cap = cv2.VideoCapture("Videos/bicycle.mp4")

model = YOLO('Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "cat", "dog", "cup", "horse", "orange",
              "traffic light", "banana", "telephone", "sofa", "chair", "cake", "mouse", "laptop", "bowl", "bird", "remote",
              "fire hydrant", "stop sign", "parking meter", "bench", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sport ball", "kite", "baseball hat", "fork",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "knife", "spoon", "apple", "sandwich",] 

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #KUTULAR İÇİN
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            #DOĞRULUK ORANLARI İÇİN
            conf = math.ceil((box.conf[0]*100))/100
            
            #SINIF İSİMLERİ İÇİN
            cls = int(box.cls[0])
                    
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)
    
    cv2.imshow('Image', img)
    cv2.waitKey(1)