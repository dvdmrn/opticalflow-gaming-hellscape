import numpy as np
import cv2 as cv
import math
np.set_printoptions(threshold=np.inf)

cap = cv.VideoCapture(cv.samples.findFile("MarioKart.mkv"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255



width =  int(cap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # float

origin = (width//2, height//2)

def threshold(x):
    if((x**3) > 720):
        return 720
    elif((x**3) < -720):
        return -720
    else:
        return x**3
    # return x**3

thresholdVec = np.vectorize(threshold)


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
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 4, 15, 3, 5, 1.2, 0)
    flow_weighted = thresholdVec(flow)
    global_ave = [np.mean(flow_weighted[...,0]),np.mean(flow_weighted[...,1])]
    print("global ave: ",global_ave)
    # print(np.mean(np.mean(flow,axis=0),axis=0))

    main_window = "global motion vec"
    # Create black empty images
    size = height, width, 3
    # atom_image = np.zeros(size, dtype=np.uint8)
    main_image = frame2

    #  2.c. Create a few lines
    my_line(main_image, origin, tuple([math.ceil(origin[0]+global_ave[0]),math.ceil(origin[1]+global_ave[1])]))
    cv.imshow(main_window, main_image)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points

    # print("printing ofa output")
    # file = open("output.txt","w")
    # file.writelines(str(np.mean(np.mean(flow,axis=0),axis=0)))
    # file.close()
    # print("file closed")
    # break

    # print()

    # # print("--------------------------------------------")
    # mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    # cv.imshow('frame2',bgr)
    # k = cv.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # elif k == ord('s'):
    #     cv.imwrite('opticalfb.png',frame2)
    #     cv.imwrite('opticalhsv.png',bgr)
    # prvs = next