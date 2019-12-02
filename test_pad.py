import numpy as np
import cv2 as cv
# np.set_printoptions(threshold=np.inf)

# cap = cv.VideoCapture(cv.samples.findFile("head_motion.mp4"))
# ret, frame1 = cap.read()
# prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[...,1] = 255



# width =  int(cap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # float

# origin = (width//2, height//2)

# # def threshold(x):
# # 	return 1/(1+np.exp(-10*(x)))

# def threshold(x):
# 	return x**3

# def printOutput(out):
# 	print("printing ofa output")
# 	file = open("output.txt","w")
# 	file.writelines(str(out))
# 	file.close()
# 	print("file closed")
# 	exit()

# while(1):
#     ret, frame2 = cap.read()
#     next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
#     flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 1, 15, 3, 5, 1.2, 0)


#     mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
#     # print(np.mean(ang)*180/np.pi/2)
#     print("mean x,y vec",threshold(np.mean(flow[...,0])),threshold(np.mean(flow[...,1])))
#     hsv[...,0] = ang*180/np.pi/2
#     hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
#     bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
#     cv.imshow('frame2',bgr)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv.imwrite('opticalfb.png',frame2)
#         cv.imwrite('opticalhsv.png',bgr)
#     prvs = next

# a = np.array([])
a = np.array([[8,8]])

b = np.array([8, 9])
print(a-b)
print(a)

a = np.append(a,[[69,69]],axis=0)
print(a)
# a=np.concatenate(a,np.array([1,1]))
# print(a)
# a=np.append(a,np.array([2,3]))
# print(a)
