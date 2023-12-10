import numpy as np 

def get_window(window):

    return


#optical flow using Weighted Least Squares 
def optical_flow_WLS(I1,I2,window_size):

    #define arrays to store optical flow vector --> (u,v) =(δx/δt, δy/δt)
    u = np.zeros_like(I1)
    v = np.zeros_like(I2)

    #iterate through each pixel 

    return 




if __name__ == '__main__':
    #defining iamges 
    path_imgt1 = './data/external/IMG_4110.png'
    path_imgt2 = './data/external/IMG_4111.png'
    #define window_size for optical flow calculations 
    wind_size = 5
    #optical flow 
    optical_flow_WLS(path_imgt1,path_imgt2,wind_size)


    pass