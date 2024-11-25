import cv2 
import numpy as np1

def plotLineLow(mask, x0, y0, x1, y1):
    num_color_pixels = 0
    num_pixels = 0

    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0

    for x in range(x0, x1+1):
        if (mask[x,y] != 0):
            num_color_pixels += 1
        
        if D > 0: 
            y = y + yi
            D = D + (2 * (dy - dx))
        else : 
            D = D + 2 * dy 

        num_pixels += 1

    ratio = num_color_pixels / num_pixels
    return ratio
        

def plotLineHigh(mask, x0, y0, x1, y1):
    num_color_pixels = 0
    num_pixels = 0

    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0

    for y in range(y0, y1+1):
        if (mask[x,y] != 0):
            num_color_pixels += 1
            
        num_pixels += 1
        
        if D > 0: 
            x = x + xi
            D = D + (2 * (dx - dy))
        else : 
            D = D + 2 * dx


    ratio = num_color_pixels / num_pixels
    return ratio


def calculate_ratio_of_color_pixels_between_two_points(mask,point1, point2):
    x0, y0 = point1
    x1, y1 = point2
    

    # Bresenham's line algorithm
    if (abs(y1 - y0) < abs(x1 - x0)):
        if (x0 > x1):
            ratio = plotLineLow(mask, x1, y1, x0, y0)
        else: 
            ratio = plotLineLow(mask, x0, y0, x1, y1)
    else :
        if (y0 > y1):
            ratio = plotLineHigh(mask, x1, y1, x0, y0)
        else: 
            ratio = plotLineHigh(mask, x0, y0, x1, y1)

    return ratio



def is_legit(mask, point1, point2, threshold):
    ratio = calculate_ratio_of_color_pixels_between_two_points(mask, point1, point2)
    """ The larger the ratio is, the more points will be connected
    since it allows more black pixels between two points """

    if ratio >= threshold:
        is_legit = True
    else: 
        is_legit = False

    return is_legit


def connect_2_legit_points(mask, point1, point2, threshold) -> np.array:
    legit, ratio = is_legit(mask, point1, point2, threshold)

    if (legit):
        cv2.line(mask, (point1[0], point1[1]), (point2[0], point2[1]), (255,255,0), 1)

    return legit