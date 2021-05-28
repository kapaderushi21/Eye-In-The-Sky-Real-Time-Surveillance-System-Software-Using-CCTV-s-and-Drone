import cv2, time, pandas
from datetime import datetime
import threading
first_frame=None



class VideoCamera(object):
    
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)
      
        # Initialize video recording environment
        self.is_record = False
        self.out = None

    
    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        ret, frame1 = self.cap.read()
        ret, frame2 = self.cap.read()
 
        while self.cap.isOpened():
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, thresh = cv2.threshold(blur, 20, 255,cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh,  None, iterations=4)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

                if cv2.contourArea(contour) < 1200:
                    continue
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame1,"Status : {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if ret:
                ret, jpeg = cv2.imencode('.jpg', frame1)
                frame1=frame2
                ret, frame2 = self.cap.read()
                return jpeg.tobytes()
      
            else:
                return None