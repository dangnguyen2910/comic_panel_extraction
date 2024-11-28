import cv2 
import numpy as np 

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
        # print((x,y), mask[x,y])
        is_white = (mask[y-2:y+3, x-2:y+3] != 0).any()
        if is_white:
            num_color_pixels += 1
        
        if D > 0: 
            y = y + yi
            D = D + (2 * (dy - dx))
        else : 
            D = D + 2 * dy 

        num_pixels += 1

    # print(f"Number of color pixels: {num_color_pixels}")
    # print(f"Number of pixels: {num_pixels}")
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
        is_white = (mask[y-2:y+3, x-2:x+3] != 0).any()
        if is_white:
            num_color_pixels += 1
            
        num_pixels += 1
        
        if D > 0: 
            x = x + xi
            D = D + (2 * (dx - dy))
        else : 
            D = D + 2 * dx

    # print(f"Number of color pixels: {num_color_pixels}")
    # print(f"Number of pixels: {num_pixels}")
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
    """ The larger the ratio is, the more points will be connected
    since it allows more black pixels between two points """

    ratio = calculate_ratio_of_color_pixels_between_two_points(mask, point1, point2)
    if ratio >= threshold:
        is_legit = True
    else: 
        is_legit = False

    return is_legit




def connect_2_legit_points(mask:np.array, target: np.array,  point1, point2, threshold) -> None:
    legit = is_legit(mask, point1, point2, threshold)

    if (legit):
        # print(legit)
        # print(f"(x0, y0) = {(point1[0], point1[1])}")
        # print(f"(x1, y1) = {(point2[0], point2[1])}")
        cv2.line(target, (point1[0], point1[1]), (point2[0], point2[1]), (255,255,0), 3)
