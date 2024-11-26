import os
from itertools import combinations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from clustering import cluster
from line import Line
from morphology import smooth_edges, morphing
from bresenham import connect_2_legit_points
from intersections import find_intersection



def classify_line(lines):
    vertical_lines = []
    horizontal_lines = []

    if lines is None: 
        return [], []

    for l in lines:
        line = Line(l[0][0], l[0][1], l[0][2], l[0][3])

        if line.orientation == 'vertical':
            vertical_lines.append(line)

        if line.orientation == 'horizontal':
            horizontal_lines.append(line)

    return vertical_lines, horizontal_lines



def create_image_with_hough_lines(image, lines):
    height, width = image.shape
    binary_line_image = np.copy(image) * 0
    line_image = np.zeros((height, width, 3))

    if lines is None: 
        return binary_line_image, line_image

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(binary_line_image, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.line(line_image, (x1, y1), (x2, y2), (0,255,255), 2)
        # cv2.circle(line_image, (x1,y1), 5, (255,0,0), -1)
        # cv2.circle(line_image, (x2, y2), 5, (255,0,0), -1)

    return binary_line_image, line_image   
 


def create_image_with_classified_lines(image, vertical_lines, horizontal_lines,is_binary=True):
    height, width = image.shape

    if (is_binary):
        line_image = np.zeros((height, width, 3))

    for line in horizontal_lines: 
        cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (255,0,0), 2)
        # cv2.circle(line_image, (line.x1, line.y1), 2, (255,255,0), -1)
        # cv2.circle(line_image, (line.x2, line.y2), 2, (255,255,0), -1)

    for line in vertical_lines:
        cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (0,0,255), 2)
        # cv2.circle(line_image, (line.x1, line.y1), 2, (0,255,255), -1)
        # cv2.circle(line_image, (line.x1, line.y1), 2, (0,255,255), -1)

    return line_image   



def distance(point1, point2):
    x0, y0 = point1
    x1, y1 = point2
    return np.sqrt((x0 - x1) ** 2 + (y0 - y1)**2)



def remove_black_noise_with_contours(image, area_threshold=10, min_width=10, min_height=10):
    inverted = (image == 0).astype(np.uint8)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_mask = np.zeros_like(image, dtype=np.uint8)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > area_threshold and w > min_width and h > min_height:
            
            cv2.drawContours(output_mask, [contour], -1, 0, thickness=cv2.FILLED)
        else:
            
            cv2.drawContours(output_mask, [contour], -1, 255, thickness=cv2.FILLED)

    
    output_image = np.maximum(image, output_mask)

    return output_image


def get_border_lines(image_shape):
    height, width = image_shape[:2]

    border_lines = [
        Line(0, 0, width - 1, 0),         # Top 
        Line(0, height - 1, width - 1, height - 1),  # Bottom 
        Line(0, 0, 0, height - 1),        # Left 
        Line(width - 1, 0, width - 1, height - 1)    # Right 
    ]

    return border_lines


def main() -> None: 
    dir = '../../../output/unet_panel/raw/'
    # img_path = "../../../output/unet_panel/raw/pred_mask_raw_1.png"
    # img_path = "../../../output/unet_panel/raw/pred_mask_raw_2.png"
    # img_path = "../../../output/unet_panel/raw/pred_mask_raw_7.png"
    img_path = "../../../output/unet_panel/raw/pred_mask_raw_13.png"
    # img_path = "../../../output/unet_panel/raw/pred_mask_raw_27.png"


    # for img_name in os.listdir(dir):
        # img_path = os.path.join(dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(figsize=(12,12))


    # Original image
    ax = fig.add_subplot(3,3,1)
    plt.imshow(img, 'gray')
    plt.title("Original mask")


    # Image after smooth edges
    ax = fig.add_subplot(3,3,2)
    smooth = smooth_edges(img)
    plt.imshow(smooth, 'gray')
    plt.title("After smooth edges")


    # Canny detection
    ax = fig.add_subplot(3,3,3)
    canny = cv2.Canny(smooth, threshold1=100, threshold2=200)
    plt.imshow(canny, 'gray')
    plt.title("After canny")


    # Erosion and dilation
    ax = fig.add_subplot(3,3,4)
    morph = morphing(canny)
    plt.imshow(morph, 'gray')
    plt.title("After erosion + dilation")

    # Remove noise
    # fig.add_subplot(3,3,5)
    # morph_no_noise = 255 - remove_black_noise_with_contours(255-morph, area_threshold=25)
    # plt.imshow(morph_no_noise, 'gray')

    # Hough transform 1st time
    ax = fig.add_subplot(3,3,5)
    lines = cv2.HoughLinesP(morph, 1, np.pi/180, threshold=23, 
                            minLineLength=50, maxLineGap=40)
    
    hough_binary, hough = create_image_with_hough_lines(img,lines)
    
    plt.imshow(hough)
    plt.title("After hough transform 1st time")


    
    # Classify lines to horizontal or vertical
    ax = fig.add_subplot(3,3,6)
    vertical_lines, horizontal_lines = classify_line(lines)
    hough_classified = create_image_with_classified_lines(img, vertical_lines, horizontal_lines)
    
    plt.imshow(hough_classified.astype(int))

    border_lines = get_border_lines(hough_binary.shape)
    for line in border_lines: 
        if line.orientation == 'vertical': 
            vertical_lines.append(line)
        else: 
            horizontal_lines.append(line)

    

    # Find intersection
    ax = fig.add_subplot(3,3,7)
    intersections = find_intersection(vertical_lines, horizontal_lines)
    plt.imshow(hough_classified.astype(int))

    for point in intersections: 
        try: 
            plt.plot(point[0], point[1], 'go', markersize=2)
        except TypeError: 
            continue 


    # Cluster points
    ax = fig.add_subplot(3,3,8)
    centroids = cluster(hough_classified, intersections)
    plt.imshow(hough_classified)

    # Connect two points that are legits
    ax = fig.add_subplot(3,3,9)
    for i, (point1, point2) in enumerate(combinations(centroids, 2)): 
        legit = connect_2_legit_points(hough_binary, point1, point2, threshold=0)

    plt.imshow(hough_binary, 'gray')
    plt.show()



if __name__ == "__main__":
    main()
