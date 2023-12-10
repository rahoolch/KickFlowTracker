import numpy as np 
import cv2 as cv 

#function to return the magnitude 
def perform_optical_flow(prev_img,img):
    #computes the dense optical flow using Farneback method 
    flow = cv.calcOpticalFlowFarneback(prev_img,img,None,0.5, 1, 10, 3, 5, 1.1, 0)

    #computes the magnitude and angle for the 2D vectors
    mag,angle = cv.cartToPolar(flow[...,0],flow[...,1])

    return mag, angle

def visualize_optical_flow(mag,angle,shape):
    #define empty mask 
    mask = np.zeros(shape,dtype=np.uint8)
    #set saturation of mask to max
    mask[...,1] = 255
    #sets image hue based on optical flow direction 
    mask[...,0] = angle * 180 / np.pi / 2
    #sets image value based on optical flow magnitude 
    mask[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    #convert hsv to rgb
    return cv.cvtColor(mask, cv.COLOR_HSV2BGR)



if __name__ == '__main__':
    #defining iamges 
    path_imgt1 = './data/external/optFlowTest/cal_IMG_28.png'
    path_imgt2 = './data/external/optFlowTest/cal_IMG_29.png'
    #reading images 
    imgt1 = cv.imread(path_imgt1)
    imgt2 = cv.imread(path_imgt2)
    #shape 
    shape = imgt1.shape
    #grayscale 
    imgt1 = cv.cvtColor(imgt1,cv.COLOR_BGR2GRAY)
    imgt2 = cv.cvtColor(imgt2,cv.COLOR_BGR2GRAY)
    #perform optical flow
    mag,angle = perform_optical_flow(imgt1,imgt2)
    #visualizing optical flow 
    opt_flow_viz = visualize_optical_flow(mag,angle,shape)
    #display 
    cv.imshow('Optical Flow',opt_flow_viz)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    pass