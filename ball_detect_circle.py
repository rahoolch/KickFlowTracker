import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

frame = cv.imread('tracking_frame.png')
grayFrame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
cv.imwrite("GrayScaleImage.png",grayFrame)
blurFrame = cv.GaussianBlur(grayFrame,(9,9),0)
cv.imwrite('BlurFrame.png',blurFrame)

circles = cv.HoughCircles(blurFrame,cv.HOUGH_GRADIENT,1,30,param1=60, param2=30, minRadius=25, maxRadius=90)

if circles is not None:
    circles = np.uint16(np.around(circles))
    # print(circles.shape)
    circles = circles[0]
    for (x,y,r) in circles:
        
        if y>500 and x>200:
            print(x,y,r)
            cv.rectangle(frame, (x-r,y-r), (x+r, y+r), (0,255,0), 2)
            # cv.circle(frame,(x,y),r,(0,255,0), 2)
            cv.imshow('circles',frame)
            cv.waitKey(0)


# print(circles[0])




# cv.destroyAllWindows()