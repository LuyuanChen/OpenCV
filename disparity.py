import numpy as np
import cv2
from matplotlib import pyplot as plt

IMAGE_DIR = './images/'
LEFT_IMAGE_DIR = IMAGE_DIR + 'left/'
RIGHT_IMAGE_DIR = IMAGE_DIR + 'right/'

image_name = '000116.jpg'

imgL = cv2.imread(LEFT_IMAGE_DIR + image_name, 0)
imgR = cv2.imread(RIGHT_IMAGE_DIR + image_name, 0)

stereo = cv2.StereoBM_create(96, 15)
disparity = stereo.compute(imgL, imgR)

plt.imshow(disparity, 'gray')
plt.show()
