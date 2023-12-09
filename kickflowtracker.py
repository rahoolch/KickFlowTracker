#import libraries
import cv2 as cv 



#entry into kickflow tracker
def main(video_input):
    #define CV video capture obj 
    video = cv.VideoCapture(video_input)
    #get fps of video feed 
    fps = int(video.get(cv.CAP_PROP_FPS))

    #check if video is unable to be obtained
    if video.isOpened() == False:
        raise Exception('Error opening video stream') 

    #ball tracking 
    while True:
        #read current frame 
        ret,frame = video.read()
        #if successfully reading frame
        if ret: 

            #preprocessing (REPLACE AND REDO MANUALLY IF NEEDED)

            #display 
            cv.imshow('KickFlow Tracker',frame)

            #quit loop if 'q' pressed
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        #quit if frame not returned 
        else:
            raise Exception('Frame not returned')
    
    #stop and release video capture object 
    video.release()
    #close all windows
    cv.destroyAllWindows()
    
    return  

if __name__ == "__main__":
    video_input = 'data/external/game_clip.mp4'
    main(video_input)