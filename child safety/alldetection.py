from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import winsound
import datetime

videolink='../videos/allinonev5.mp4'
cap=cv2.VideoCapture(videolink)

vcap = cap
width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count=0
model=YOLO('../yolo-weights/bestn.pt')
modelweapon = YOLO("../yolo-weights/bestweapon.pt")

my_dict={}

start_time = datetime.datetime.now()
start_minute = start_time.second
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

# Tracker
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
countweapon=0
limitsup=[2,2,width-10,2]
limitsdown=[2,height-10,width-10,height-10]
personinitial=[]
personfall=[]

totalCountsup=[]
totalCountsdown=[]
exitid=[]
classNames2=['pistol']
while(True):
    success , img=cap.read()
    imgRegion=cv2.bitwise_and(img,img)
    results=model(imgRegion,stream=True)
    resultweapon=modelweapon(imgRegion,stream=True)
    detections=np.empty((0,5))
    for r in resultweapon:
        boxes=r.boxes
        for box in boxes:
            myColor = (0,255, 0)
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            cv2.rectangle(img,(x1,y1),(x2,y2),myColor,3)
            conf=math.ceil(box.conf[0]*100)/100 #confidence
            # print(conf)
            cls=int(box.cls[0])
            if(classNames2[cls]=='pistol' and conf>0.5):
                myColor=(0,0,255)
                print("HIIIII")
                print("HIIIII")
                
                cvzone.putTextRect(img, f'pistol', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
                filename="Image"+str(countweapon)+".jpg"
                countweapon+=1
                cv2.imwrite(filename,img)
                winsound.Beep(1445, 100)


    for r in results:
        boxes=r.boxes
        count2=0
        for box in boxes:

            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            confidence=box.conf[0]
            confidence=math.ceil((box.conf[0]*100))/100
            print(confidence)

            cls=int(box.cls[0])
            currentClass=classNames[cls]
            if(currentClass=="person" or currentClass=="ride" and confidence>0.5):
                count2+=1
                # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9,rt=5)
                # cvzone.putTextRect(img,f'{classNames[cls]} {confidence}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                currentArray=np.array([x1,y1,x2,y2,confidence])
                detections=np.vstack((detections,currentArray))

    resultsTracker=tracker.update(detections)
    cv2.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,0,255),4)
    cv2.line(img,(limitsdown[0],limitsdown[1]),(limitsdown[2],limitsdown[3]),(0,0,255),4)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 ,id= int(x1), int(y1), int(x2), int(y2),int(id)
        print(result)
        cvzone.cornerRect(img,(x1,y1,x2-x1,y2-y1),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=1,
                            offset=3)
        #     finding center
        cx,cy=x1+(x2-x1)//2,y1+(y2-y1)//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)


        if x2-x1>1.25*(y2-y1):
            curr_time = datetime.datetime.now()

            if(my_dict.get(id) is not None and (curr_time-my_dict[id]).seconds>5):
                myColor = (0, 0, 255)
                cvzone.putTextRect(img, f'person just fell ', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
                filename = "Image1"+str(count)+".jpg"
                cv2.imwrite(filename, img)
                winsound.Beep(1445, 100)
                count+=1
        else:
            my_dict[id]=datetime.datetime.now()
        if personinitial.count(id)==0:
            personinitial.append(id)
            
    if count2 > 15:
        filename = "Image" + str(filecount) + ".jpg"
        filecount += 1
        cv2.imwrite(filename, img)
        print("Exceeded Limit\n")
        print("Exceeded Limit\n")
        print("Exceeded Limit\n")
        print("Exceeded Limit\n")
        winsound.Beep(1444,100)
        cvzone.putTextRect(img, f'Count Exceeded !! {count2} !!', (50, 50), scale=2, thickness=1,offset=3)
    else:
        cvzone.putTextRect(img, f'Counts {count2} : ', (50, 50), scale=2, thickness=1,offset=3)


    cv2.imshow("Image",img)
    # cv2.imshow("Imageregion", imgRegion)
    cv2.waitKey(1)
