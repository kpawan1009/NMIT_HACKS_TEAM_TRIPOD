import pyttsx3
import winsound
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap=cv2.VideoCapture('../videos/Children-restricted-area.mp4')
model=YOLO('../yolo-weights/bestn.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
maskblack=cv2.imread("masks/childRestrictblackmask.png")
tobeshown=cv2.imread("masks/childRestrictredmask.png")
count2=0
# Tracker
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
count=0

while(True):
    success , img=cap.read()
    imgRegion=cv2.bitwise_and(img,maskblack)
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:

            # Bounding Box

            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)

            # confidence

            confidence=box.conf[0]
            confidence=math.ceil((box.conf[0]*100))/100;
            print(confidence)

            cls=int(box.cls[0])
            currentClass=classNames[cls]
            if(currentClass=="person" and confidence>0.3):
                if count>7 and count<12:
                    text_speech = pyttsx3.init()
                    text_speech.say("Entry is prohibited")
                    text_speech.runAndWait()
                    count2+=1
                    # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9,rt=5)
                currentArray=np.array([x1,y1,x2,y2,confidence])
                detections=np.vstack((detections,currentArray))
                cvzone.putTextRect(img, f'Entry Not allowed - Image Captured ',(25, 25), scale=1, thickness=1)
                filename = "Image" + str(count) + ".jpg"
                cv2.imwrite(filename, img)
                count2+=1
                winsound.Beep(1445, 100)
                count+=1

        #     finding center
            cx,cy=x1+(x2-x1)//2,y1+(y2-y1)//2
            cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

    cv2.imshow("Image",cv2.bitwise_and(img,tobeshown))
    # cv2.imshow("Imageregion", imgRegion)
    cv2.waitKey(1)