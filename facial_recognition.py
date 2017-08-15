import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import sys
import time


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

mouth_cascade = cv2.CascadeClassifier('mouth.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
e2fRatio_min = 0.2142
e2fRatio_max = 0.5
f2pRatio_min = .1
f2pRatio_max = 0.999
f2mRatio_min = 0.245
f2sRatio_min = 0.245

#print sys.argv[1]
img = cv2.imread(sys.argv[1])
#video_capture = cv2.VideoCapture(0)
img = cv2.resize(img, (240, 320))
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

i =0
while i==0:
    # FACES
    #rawCapture = PiRGBArray(camera)

    # allow the camera to warmup
    time.sleep(0.3)

    # grab an image from the camera
    #camera.capture(rawCapture, format="bgr")
    #img = rawCapture.array
    #img = cv2.resize(img, None, fx=0.25, fy=0.25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    pheight, pwidth, pchannels = img.shape 
    print "img dim {0}:{1}".format(pheight,pwidth)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1, minSize=(
    int(pheight * f2pRatio_min), int(pwidth * f2pRatio_min)),maxSize=(int(pheight * f2pRatio_max), int(pwidth * f2pRatio_max)),  flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    print "{} faces found".format(len(faces))

    for (x, y, w, h) in faces:
    
        print "face dim = {}".format((w, h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
    
       # EYES
        
        eyes = eye_cascade.detectMultiScale(roi_gray, minSize=(int(e2fRatio_min * w), int(e2fRatio_min * h)), maxSize=(int(e2fRatio_max * w), int(e2fRatio_max * h)))
        print "{} eyes found".format(len(eyes))
    
        for (ex, ey, ew, eh) in eyes[0:2]:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            print "eye dim = {}".format(ew, eh)
       
    
      # MOUTHS
      
        mouths = mouth_cascade.detectMultiScale(roi_gray, minSize=(int(f2mRatio_min * w),int(f2mRatio_min *h)))
        print "{} mouths found".format(len(mouths))
        if len(mouths)>0:
            mx, my, mw, mh=mouths[-1]
            #cv2.putText(roi_color,str(i),(int(mx+mw/2),int(my+mh/2)), font, 1,(255,255,255),2)
            print "mouth size, pos = ([{0}:{1}]) | ({2},{3}".format(mw, mh, mx, my)
            if my <= ey + eh:
                print "no mouths in hindsight"
            else:
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
        else:
            mx,my,mw,mh=0,0,0,0
        
    
       # SMILES
    
        smiles=smile_cascade.detectMultiScale(roi_color,minSize=(int(f2sRatio_min * w), int(f2sRatio_min * h)))
        print len(smiles), "Smiles Detected"
        if len (smiles) >0:
            sx,sy,sw,sh=smiles[-1]
            print "Dims of smile = {0},{1}".format(sw,sh)
            if sy <= ey + eh:
                print "no smiles in hindsight"
            else:
                cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
    
        else:
            sx,sy,sw,sh=0,0,0,0
    time.sleep(1)
    i=1
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
