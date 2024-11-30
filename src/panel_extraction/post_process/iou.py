import numpy as np
from skimage.measure import label

def calculate_iou(pred_binary, gt_binary):
    """
    Compute IoU for two binary masks.
    
    Parameters:
    -----------
    pred_binary (numpy array): Binary mask for predicted object.
    gt_binary (numpy array): Binary mask for ground truth object.
    
    Returns:
    --------
    float: IoU value.
    """

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    return intersection / union if union > 0 else 0



def calculate_mean_iou(predicted_mask, ground_truth_mask):
    """
    Calculate Mean IoU (mIoU) for binary masks with multiple objects.

    Parameters:
    -----------
    predicted_mask (numpy array): Binary mask from the model.
    ground_truth_mask (numpy array): Binary ground truth mask.

    Returns:
    --------
    float: Mean IoU value.
    list: IoU values for each ground truth object.
    """

    pred_labels = label(predicted_mask)
    gt_labels = label(ground_truth_mask)
    
    pred_objects = np.unique(pred_labels[pred_labels > 0])
    gt_objects = np.unique(gt_labels[gt_labels > 0])
    
    iou_values = []

    for gt_label in gt_objects:
        gt_binary = gt_labels == gt_label
        best_iou = 0
        
        for pred_label in pred_objects:
            pred_binary = pred_labels == pred_label
            iou = calculate_iou(pred_binary, gt_binary)
            best_iou = max(best_iou, iou)
        
        iou_values.append(best_iou)
    
    mean_iou = np.mean(iou_values) if iou_values else 0
    return mean_iou, iou_values


