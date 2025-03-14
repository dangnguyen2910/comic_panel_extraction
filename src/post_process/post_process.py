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
from metric import calculate_mean_iou, calculate_mean_dice,\
                   calculate_iou, calculate_dice
from connect_centroids import draw_centroid_connection



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
    line_image = np.zeros((height, width, 3), dtype=np.uint8)
    res = image.copy()

    if lines is None: 
        return res, binary_line_image, line_image

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(binary_line_image, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.line(line_image, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.line(res, (x1, y1), (x2, y2), (0,0,0), 2)

    return res, binary_line_image, line_image   
 


def create_image_with_classified_lines(image, vertical_lines, horizontal_lines):
    height, width = image.shape

    line_image = np.zeros((height, width, 3))

    for line in horizontal_lines: 
        cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (255,0,0), 2)

    for line in vertical_lines:
        cv2.line(line_image, (line.x1, line.y1), (line.x2, line.y2), (0,0,255), 2)

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
    # dir = '../../../output/unet_panel/vagabond'
    dir = '../../../output/unet_panel/mixed'
    pred_dirs = []

    # Vagabond
    # pred_dirs.append('line_connected_threshold75/')
    # pred_dirs.append('raw_filter/')

    # Mixed 
    # pred_dirs.append('new_pred_mask_raw_val/')
    pred_dirs.append('new_pred_mask_raw/')
    # pred_dirs.append('new_line_connected_Enoise_val')
    # pred_dirs.append('new_line_connected_Enoise_test')


    gt_dir = os.path.join(dir, 'new_true_mask')
    print(f"Using gt: {os.path.basename(gt_dir)}")

    for pred_dir in pred_dirs: 
        ious = []
        dices = []
        pred_dir_path = os.path.join(dir, pred_dir)
        print('-' * 50)
        print(f"Calculate metric for {pred_dir}")

        for m,img_name in enumerate(os.listdir(pred_dir_path)):
            img_path = os.path.join(pred_dir_path, img_name)
            gt_path = os.path.join(gt_dir, 'true_mask_' + img_name[14:])

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)


            # smooth = smooth_edges(img)

            # # Canny detection
            # canny = cv2.Canny(smooth, threshold1=100, threshold2=200)

            # # Erosion and dilation
            # morph = morphing(canny)

            # # Remove noise
            # remove_noise = True
            # if remove_noise:
            #     morph = 255 - remove_black_noise_with_contours(255-morph, area_threshold=25)


            # # Hough transform 1st time
            # lines = cv2.HoughLinesP(morph, 1, np.pi/180, threshold=23, 
            #                         minLineLength=50, maxLineGap=60)
            
            # res, hough_binary, hough = create_image_with_hough_lines(img,lines)


            # # Classify lines to horizontal or vertical
            # vertical_lines, horizontal_lines = classify_line(lines)
            # hough_classified = create_image_with_classified_lines(img, vertical_lines, horizontal_lines)
            

            # border_lines = get_border_lines(hough_binary.shape)
            # for line in border_lines: 
            #     if line.orientation == 'vertical': 
            #         vertical_lines.append(line)
            #     else: 
            #         horizontal_lines.append(line)

            

            # # Find intersection
            # intersections = find_intersection(vertical_lines, horizontal_lines)
            # # for point in intersections: 
            # #     plt.plot(point[0], point[1], 'go', markersize=2)


            # # Cluster points
            # centroids = cluster(hough_classified, intersections)
            # centroids = centroids.astype("int")

            # # Connect two points that are legits
            # # for i, (point1, point2) in enumerate(combinations(centroids, 2)): 
            # res = draw_centroid_connection(hough_binary, res, centroids)

            mean_iou, _ = calculate_mean_iou(img, gt)
            mean_dice, _ = calculate_mean_dice(img, gt)
            dices.append(mean_dice)
            ious.append(mean_iou)

            # fig = plt.figure(figsize=(12,12))
            # fig.add_subplot(1,2,1)
            # plt.imshow(img, 'gray')
            # plt.title(f"Final mask\nmIoU: {mean_iou:.3f}\nMean dice coef: {mean_dice:.3f}")
            # fig.add_subplot(1,2,2)
            # plt.imshow(gt, 'gray')
            # plt.title(f"Ground truth")
            # plt.show()

        
        print(f"mIoU: {np.mean(ious):.3f}")
        print(f"Mean dice coef: {np.mean(dices):.3f}")



if __name__ == "__main__":
    main()
