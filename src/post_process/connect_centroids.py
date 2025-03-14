import numpy as np
import cv2 

def bresenham(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points

def is_connection_valid(image, x1, y1, x2, y2, threshold=0.75):
    line_points = bresenham(x1, y1, x2, y2)
    colored_count = 0

    for x, y in line_points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  
            if np.any(image[y, x] > 20):  
                colored_count += 1

    return (colored_count / len(line_points)) >= threshold

def close(image):
    # gaussian_ksize = (7, 7) 
    
    dilation_kernel_1 = np.ones((3, 3), np.uint8)
    erosion_kernel_1 = np.ones((5, 1), np.uint8)
    dilated_image = cv2.dilate(image, dilation_kernel_1, iterations=1)
    eroded_image_2 = cv2.erode(dilated_image, erosion_kernel_1, iterations=1)
    #dilated_image_2 = cv2.dilate(eroded_image_2, dilation_kernel_2, iterations=1)
#     processed_image = cv2.erode(eroded_image_2, erosion_kernel_2, iterations=1)
    
    return eroded_image_2


def draw_centroid_connection(hough_classified1,image1,centroids):
    # image1copy = close(hough_classified1)
    valid_connections = []
    ve_len = image1.copy()

    for i, point1 in enumerate(centroids):
        for j, point2 in enumerate(centroids):
            if i != j:  
                x1, y1 = int(point1[0]), int(point1[1])
                x2, y2 = int(point2[0]), int(point2[1])
            
                if is_connection_valid(hough_classified1, x1, y1, x2, y2):
                    if np.sqrt((x1-x2)**2 + (y1-y2)**2) > 40:
                        cv2.line(ve_len, (x1, y1), (x2, y2), (0), 5)

    return ve_len 