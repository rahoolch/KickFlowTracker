#import libraries
import cv2 as cv 
import motion_utils as mu 



#entry into kickflow tracker
def main(video_input):
    #define CV video capture obj 
    video = cv.VideoCapture(video_input)
    #get fps of video feed 
    fps = int(video.get(cv.CAP_PROP_FPS))

    #check if video is unable to be obtained
    if video.isOpened() == False:
        raise Exception('Error opening video stream') 
    
    #get first frame 
    ret,first_frame = video.read()
    #define shape 
    shape = first_frame.shape 
    #init prev_img 
    prev_img = cv.cvtColor(first_frame,cv.COLOR_BGR2GRAY)

    #ball tracking 
    while True:
        #read current frame 
        ret,frame = video.read()
        #if successfully reading frame
        if ret: 
            #convert cur frame to grayscale 
            frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            #optical flow 
            mag,angle = mu.perform_optical_flow(prev_img,frame)
            #optical flow visualization
            opt_flow_viz = mu.visualize_optical_flow(mag,angle,shape)

            #display 
            cv.imshow('KickFlow Tracker',opt_flow_viz)

            #quit loop if 'q' pressed
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            #update prev_img
            prev_img = frame
        
        #quit if frame not returned 
        else:
            raise Exception('Frame not returned')
    
    #stop and release video capture object 
    video.release()
    #close all windows
    cv.destroyAllWindows()
    
    return  

if __name__ == "__main__":
    video_input = '/Users/allenlau/Documents/CCNY/Fall23/CSCI6516_ComputerVision/project/KickFlowTracker/data/external/vid5.mov'
    main(video_input)