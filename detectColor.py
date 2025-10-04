import cv2 as cv
import numpy as np

GreenLower = np.array([40, 50, 50])
GreenUpper =  np.array([90, 255, 255])

RedLower =np.array([0,150,100])
RedUpper= np.array([80,255,255])

def DetectColor(frame,Lower,Upper,COLORNAME='',RectBGR=(255,100,10)):
    perimeters=0
    detect = False
    HsvFrame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(HsvFrame,Lower,Upper)
    kernel = np.ones((5,5),np.uint8)
    Resmask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    contours, _ = cv.findContours(Resmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if  cv.contourArea(cnt) > 400:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(frame, [box], 0,RectBGR, 2)
            x,y,w,h= cv.boundingRect(cnt)
            perimeters = (w*h)#Calculate the object Perimeters
            cv.putText(frame, COLORNAME, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.4,RectBGR, 2)
            detect=True        
    return detect,perimeters


cap = cv.VideoCapture(0)
while True:
    ret, webcam = cap.read()
    GreenPillar,GreenPerimeters= DetectColor(webcam,GreenLower,GreenUpper,COLORNAME="GREEN")
    if GreenPillar:
        print("Detect Green")
    cv.imshow("WEBCAM", webcam)
    if cv.waitKey(1) == 27: 
        break
cap.release()
cv.destroyAllWindows()