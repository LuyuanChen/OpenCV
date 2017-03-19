# import the necessary packages
import argparse
import cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="./cascade_xml/cars.xml",
	help="path to car detector haar cascade")
args = vars(ap.parse_args())

# load the input image and convert it to grayscale
image_name = args["image"]

IMAGE_DIR = './images/'
LEFT_IMAGE_DIR = IMAGE_DIR + 'left/'
RIGHT_IMAGE_DIR = IMAGE_DIR + 'right/'
imgL = cv2.imread(LEFT_IMAGE_DIR + image_name, 0)
imgR = cv2.imread(RIGHT_IMAGE_DIR + image_name, 0)

# load the car detector Haar cascade, then detect cars in the input image
detector = cv2.CascadeClassifier(args["cascade"])
rects = detector.detectMultiScale(imgL, scaleFactor=1.04, #1.1
								  minNeighbors=5, minSize=(45, 45)) #minNeighbors=10, minSize=(45, 45))
# get disparity
stereo = cv2.StereoBM_create(96, 15)
disparity = stereo.compute(imgL, imgR)

# loop over the cars and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
	cv2.rectangle(imgL, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # get average disparity of the area
    

# show the detected cars
cv2.imshow("Cars", imgL)
cv2.waitKey(3)

# show the disparity
plt.imshow(disparity, 'gray')
plt.show()

