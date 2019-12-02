import cv2 as cv
import numpy as np
W = 400

def my_line(img, start, end):
    thickness = 2
    line_type = 8
    cv.line(img,
             start,
             end,
             (100, 255, 0),
             thickness,
             line_type)
main_window = "global motion vec"
# Create black empty images
size = W, W, 3
# atom_image = np.zeros(size, dtype=np.uint8)
main_image = np.zeros(size, dtype=np.uint8)

#  2.c. Create a few lines
my_line(main_image, (0, 0), (W, W))
# my_line(main_image, (W // 4, 7 * W // 8), (W // 4, W))
# my_line(main_image, (W // 2, 7 * W // 8), (W // 2, W))
# my_line(main_image, (3 * W // 4, 7 * W // 8), (3 * W // 4, W))
cv.imshow(main_window, main_image)
cv.moveWindow(main_window, W, 200)
cv.waitKey(0)
cv.destroyAllWindows()