import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from itertools import combinations
from clustering import cluster


class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.a = y2 - y1
        self.b = x1 - x2
        self.c = (y1 - y2) * x1 + (x1 - x2) * y1

        self.orientation = self.set_orientation()
        self.length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 )

    def set_orientation(self):
        angle = self.calculate_angle()
        if angle > 180: 
            angle = angle - 180
        
        if angle > 90: 
            angle = 180 - angle

        if angle >= 45:
            return "vertical"
        else:
            return "horizontal"
    
        
    def calculate_angle(self):
        ox = (1,0)
        A = (self.x1, self.y1)
        B = (self.x2, self.y2)
        AB = (self.x2 - self.x1, self.y2 - self.y1)
        mag_AB = np.sqrt(AB[0]**2 + AB[1]**2)
        dot_prod = np.dot(AB, ox)
        mag_prod = mag_AB * 1
        angle = math.acos(dot_prod / mag_prod)  
        return math.degrees(angle) % 360


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



def smooth_edges(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  # chỉnh nếu cần
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

    return closed_mask


def morphing(image):
    # gaussian_ksize = (7, 7) 
    erosion_kernel_1 = np.ones((5, 5), np.uint8)
    dilation_kernel_1 = np.ones((5, 5), np.uint8)
    erosion_kernel_2 = np.ones((3, 3), np.uint8)
    dilation_kernel_2 = np.ones((5, 5), np.uint8)

    #blurred_mask = cv2.GaussianBlur(binary_mask, gaussian_ksize, 0)
   # eroded_image = cv2.erode(image, erosion_kernel_1, iterations=1)
    dilated_image = cv2.dilate(image, dilation_kernel_1, iterations=2)
    eroded_image_2 = cv2.erode(dilated_image, erosion_kernel_1, iterations=1)
    #dilated_image_2 = cv2.dilate(eroded_image_2, dilation_kernel_2, iterations=1)
    processed_image = cv2.erode(eroded_image_2, erosion_kernel_2, iterations=1)
    
    return processed_image



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


def distance_from_point_to_line(point, line) -> float:
    x,y = point
    return abs(line.a * x + line.b * y + line.c)/np.sqrt(line.a**2 + line.b**2)


def distance_2_points(point1, point2) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2)**2 + (y1-y2)**2)


def calculate_intersection(line1, line2) -> tuple:
    x1, y1, x2, y2 = line1.x1, line1.y1, line1.x2, line1.y2
    x3, y3, x4, y4 = line2.x1, line2.y1, line2.x2, line2.y2

    denominator = (line1.x1 - line1.x2) * (line2.y1 - line2.y2) - (line1.y1 - line1.y2) * (line2.x1 - line2.x2)
    if denominator == 0:  
        return None # Parallel lines
    
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denominator
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denominator

    px = int(px)
    py = int(py)

    d11 = distance_2_points((px, py), (line1.x1, line1.x2))
    d12 = distance_2_points((px,py), (line1.x2, line1.y2))
    d21 = distance_2_points((px, py), (line2.x1, line2.y1))
    d22 = distance_2_points((px,py), (line2.x2, line2.y2))

    is_inside_line1 = (d11 + d12) <= (line1.length)
    is_inside_line2 = (d21 + d22) <= (line2.length)

    if (is_inside_line1 or is_inside_line2):
        return (px, py)
        
    return None


def find_intersection(vertical_lines, horizontal_lines) -> list[tuple]:
    point_list = []
    for vline in vertical_lines: 
        for hline in horizontal_lines:
            points = calculate_intersection(vline, hline)
            point_list.append(points)

    return point_list


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
    try: 
        x0, y0 = point1
        x1, y1 = point2
    except TypeError: 
        return None
    

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
    try: 
        ratio = calculate_ratio_of_color_pixels_between_two_points(mask, point1, point2)
        if ratio >= threshold:
            is_legit = True
        else: 
            is_legit = False

        return is_legit
    except TypeError: 
        return False

    



def connect_2_legit_points(mask, point1, point2, threshold) -> np.array:
    legit = is_legit(mask, point1, point2, threshold)

    if (legit):
        cv2.line(mask, (point1[0], point1[1]), (point2[0], point2[1]), (255,255,0), 1)

    return legit



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



def main() -> None: 
    dir = '../../../output/unet_panel/raw/'
    img_path = "../../../output/unet_panel/raw/pred_mask_raw_1.png"
    # img_path = "../../../output/unet_panel/raw/pred_mask_raw_7.png"
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
        legit = connect_2_legit_points(hough_binary, point1, point2, threshold=0.1)

    # plt.imshow(hough_binary, 'gray')
    plt.show()



if __name__ == "__main__":
    main()




"""  # Hough transform 2nd time
ax = fig.add_subplot(3,3,6)

lines = cv2.HoughLinesP(img_line, 1, np.pi/180, threshold=25, 
                        minLineLength=10, maxLineGap=80)

img_line = create_image_with_hough_lines(img, lines, is_binary=True)

plt.imshow(img_line, 'gray')
plt.title("After hough transform 2nd time") """

""" # CCL on hough line image
ax = fig.add_subplot(3,3,6)

(num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(255-   hough, 4, cv2.CV_32S)
plt.imshow(labels) """
