import cv2
import numpy as np

# Set up video capture
cap = cv2.VideoCapture('/Users/rahool/Desktop/CCNY/Semester 3/Computer_Vision/Final_project/All_videos/IMG_4151.MOV') 

# Lucas-Kanade optical flow params
lk_params = dict(winSize=(15, 15), 
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create random colors for rects    
color = np.random.randint(0,255,(100,3))

# Take first frame 
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):
    # Detect ball in frame using HSV threshold
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:

            (x, y, w, h) = cv2.boundingRect(cnt)
    
            # Set tracking point at ball center 
            prevPt = np.array([[x+w/2], [y+h/2]], np.float32)
            
            # Calculate flow at ball center
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, frame, prevPt, None, **lk_params) 
            # Draw rectangle on ball
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
    # Get flow at detected ball area
    flow = cv2.calcOpticalFlowPyrLK(prev_gray, frame, None, None, **lk_params)        
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display image with rects
    cv2.imshow('Frame', frame)      
      
    # Exit if ESC is pressed 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
            
cap.release()
cv2.destroyAllWindows()