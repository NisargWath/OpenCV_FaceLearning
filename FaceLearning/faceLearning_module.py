from re import M
import cv2
import mediapipe  as mp
import numpy as np
import time


class faceLearning:
    
    def __init__(self):

        self.mpDraw =  mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)

    def findFace(self, img):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
                                
        if self.results.multi_face_landmarks:
            faces = []
            for faceLmS in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img,faceLmS,)
                face = []   
                for id ,lm in enumerate(faceLmS.landmark):
                        # print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x+iw), int(lm.y*ih)
                    face.append([x,y])
                    print(id,x,y)
                faces.append(face)
        return img, faces

    
    
    
    
    
def  main():
     
    cap = cv2.VideoCapture("Resources/test_video2.mp4")
    pTime = 0
    detector = faceLearning()
    
    while True:
        success , img = cap.read()
        img, faces = detector.findFace(img)
        if len(faces)!=0:
            print(len(faces))
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS : {int(fps)}', (20,170),cv2.FONT_HERSHEY_TRIPLEX, 5, (255,0,0),3)

        cv2.imshow("Result", img)
        cv2.waitKey(1)
        
if __name__ == "__main__":
    main()