# The original code is from a udamy course self drivng cars 
# this code is under 'fair use'
# https://www.udemy.com/course/autonomous-cars-deep-learning-and-computer-vision-in-python/ 

import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg


source = cv2.VideoCapture('forza1.mp4') #Name the file you want to read from


while True:

    # extracting the frames
    ret, img = source.read()

    width = 1280
    height = 720
    size = (width, height)
    image_c = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
   
    image_g = cv2.cvtColor(image_c, cv2.COLOR_RGB2GRAY)
    image_blurred = cv2.GaussianBlur(image_g, (7, 7), 0)
    threshold_low = 10
    threshold_high = 200

    image_canny = cv2.Canny(image_blurred, threshold_low, threshold_high)
    # Visualize the region of interest
    vertices = np.array([[(1,366),(1,720),(253,717),(580,320),(1027,716),(1280,720),(1278,347),(661,305)]], dtype=np.int32)   #outline of whereb to look for line (improved) 
    mask = np.zeros_like(image_g)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image_g, mask)
    masked_image = cv2.bitwise_and(image_canny, mask)

    rho = 2            # distance resolution in pixels
    theta = np.pi/180  # angular resolution in radians
    threshold = 40     #40 minimum number of votes
    min_line_len = 100  #100 minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    # Create an empty black image
    line_image = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)

    if lines is not None: # to fix none error
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), [0, 0, 255], 20)
            lines

            α = 1
            β = 1
            γ = 0

            # Resultant weighted image is calculated as follows: original_img * α + img * β + γ
            Image_with_lines = cv2.addWeighted(image_c, α, line_image, β, γ)

            cv2.imshow("Live", Image_with_lines) # I need to save the Image_with_lines image to compile a video

            
            # exiting the loop
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
               
