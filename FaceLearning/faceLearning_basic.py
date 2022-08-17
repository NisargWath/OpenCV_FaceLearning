from re import M
import cv2
import mediapipe  as mp
import numpy as np
import time

cap = cv2.VideoCapture("Resources/test_video2.mp4")
pTime = 0

mpDraw =  mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)


while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLmS in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLmS,)
            
            for id ,lm in enumerate(faceLmS.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x+iw), int(lm.y*ih)
                print(id,x,y)
    

    
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (20,170),cv2.FONT_HERSHEY_TRIPLEX, 5, (255,0,0),3)

    cv2.imshow("Result", img)
    cv2.waitKey(1)