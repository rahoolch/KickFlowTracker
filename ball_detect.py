import cv2 as cv

#CAMERA_INDEX = 1 # 0 for built-in camera, 1 for external camera

cap = cv.VideoCapture('/Users/rahool/Desktop/CCNY/Semester 3/Computer_Vision/Final_project/All_videos/IMG_4152.MOV') # Initialize camera capture
tracker = cv.TrackerCSRT_create() # Initialize tracker with CSRT algorithm
BB = None # Bounding Box

def track(frame):
    (success, box) = tracker.update(frame)
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return success, frame

while True:
    ret, frame = cap.read()
    
    if BB is not None:
        success, frame = track(frame) # Track object
        
        
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF
    
    if key == ord("c"): # Select Region of Interest (ROI) to track
        cv.imwrite('tracking_frame.png',frame)
        BB = cv.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        print(BB)
        tracker.init(frame, BB)
        
    elif key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()