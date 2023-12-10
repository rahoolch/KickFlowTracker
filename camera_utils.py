#import libraries
import cv2 as cv 
import os


#function to write frames to image
def write_frames(video_input,out,fname,frame_skip):
    #write out dir if doesnt exist
    if not os.path.exists(out):
        os.makedirs(out)

    #define CV video capture obj 
    video = cv.VideoCapture(video_input)

    #check if video is unable to be obtained
    if video.isOpened() == False:
        raise Exception('Error opening video stream') 

    count = 0
    #loop through frames
    while True:
        #read current frame 
        ret,frame = video.read()

        if ret:
            if count%frame_skip==0:
                cv.imwrite(os.path.join(out,f'{fname+'_'+str(count)+'.png'}'),frame)
        else:
            break
        count+=1
    
    return  

if __name__ == "__main__":
    # input = '/Users/allenlau/Documents/CCNY/Fall23/CSCI6516_ComputerVision/project/KickFlowTracker/data/external/camera_calibration.MOV'
    # write_frames(input,'data/external/camera_calibration','cal_IMG',50)