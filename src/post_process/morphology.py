import cv2
import numpy as np 

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
