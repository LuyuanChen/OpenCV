import argparse
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def load_image(image_path):
    return mpimg.imread(image_path)

def save_plot(dir_name, name, img):
    cv2.imwrite(join(dir_name, name), img)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_noise(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = (255,)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=(255, 0, 0), thickness=1):
    right_slope = []
    left_slope = []

    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            m = ((y2-y1)/(x2-x1)) # slope
            if m < 0:
                left_slope.append(m)
                left_lines.append((x2,y2))
            else:
                right_slope.append(m)
                right_lines.append((x1,y1))
    
    right_slope = sorted(right_slope)[int(len(right_slope)/2)]
    left_slope = sorted(left_slope)[int(len(left_slope)/2)]

    left_y1 = min([line[1] for line in left_lines])
    left_pair = tuple([line[0] for line in left_lines if line[1] == left_y1] + [left_y1])

    right_y1 = min([line[1] for line in right_lines])
    right_pair = tuple([line[0] for line in right_lines if line[1] == right_y1] + [right_y1])

    left_x = int((img.shape[1]-left_pair[1])/left_slope) + left_pair[0]
    right_x = int((img.shape[1]-right_pair[1])/right_slope) + right_pair[0]
    
    cv2.line(img, left_pair, (left_x, img.shape[1]), color, thickness)
    cv2.line(img, right_pair, (right_x, img.shape[1]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                            np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)

    plotted_slopes = []  # an int (rounded) array for elliminating duplicates
    slope = 0

    for i in xrange(lines.shape[0]):
        for x1,y1,x2,y2 in lines[i]:
            slope = np.round(10*float(y2-y1)/(x2-x1))
            print "line slope:" + str(slope)
            if abs(y2-y1) > 50 and slope not in plotted_slopes:
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                plotted_slopes.append(slope)
    return line_img

def weighted_img(edges, line_img, alpha=1., beta=0.8, upsilon=0.5):
    color_edges = np.dstack((edges, edges, np.copy(edges) * 0))
    return cv2.addWeighted(color_edges, alpha, line_img, beta, upsilon)

def get_files(dir_path):
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

def process_img(name, img):
    # ref: https://github.com/paramaggarwal/CarND-LaneLines-P1/blob/master/P1.ipynb

    dir_name = "output_images"
    line_image = np.copy(img)*0
	
    gray_img = grayscale(img)
    blur_gray = gaussian_noise(gray_img, 5)
    edges = canny(blur_gray, 50, 150)

    # imshape = img.shape
    # vertices = np.array([[(110,imshape[0]),(410, 310),(480, 310), (imshape[1],imshape[0])]], dtype=np.int32)
    # masked_edges = region_of_interest(edges, vertices)
    
    lines = hough_lines(edges, 1, 5*np.pi/180, 100, 30, 100)
    final_img = weighted_img(lines, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imshow('', final_img)
    cv2.waitKey()
    # save_plot(dir_name, "final_"+name, final_img)

if __name__ == '__main__':
    img = load_image('./images/lae.jpg')
    process_img('lane', img)
