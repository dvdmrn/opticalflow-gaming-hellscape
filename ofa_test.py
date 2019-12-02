import numpy as np
import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()
cap = cv.VideoCapture(args.image)

width =  int(cap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # float

origin = (width//2, height//2)

print("width: ",width,"\nheight: ",height)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

def my_line(img, start, end):
    thickness = 2
    line_type = 4
    cv.line(img,
             start,
             end,
             (100, 255, 0),
             thickness,
             line_type)

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    print("p1: -----------\n",p1,"the mean: ",tuple(np.mean(p1,axis=0)[0]))
    


    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new, good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    # img = cv.add(frame,mask)
    # cv.imshow('frame',img)
    # k = cv.waitKey(30) & 0xff
    # if k == 27:
    #     break

    main_window = "global motion vec"
    # Create black empty images
    size = height, width, 3
    # atom_image = np.zeros(size, dtype=np.uint8)
    main_image = np.zeros(size, dtype=np.uint8)

    #  2.c. Create a few lines
    my_line(main_image, origin, tuple(np.mean(p1,axis=0)[0]))
    cv.imshow(main_window, main_image)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

# a = [[[1,1],[1,-1],[1,-5]],
#     [ [0,0],[0,-0.555],[0,0]]]

# print(np.mean(a,axis=(0)))

# print("mean of means", np.mean(np.mean(a,axis=(0)),axis=0))

# print(np.mean([1,-1,-5,-0.555,0,0]))