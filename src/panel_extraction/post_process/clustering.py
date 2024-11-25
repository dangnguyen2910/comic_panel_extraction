import numpy as np
from sklearn.cluster import MeanShift
import cv2 

def cluster(hough_classified1, intersections):
    intersections1 = [item for item in intersections if item is not None]
    intersections1 = np.array(intersections1)

    mean_shift = MeanShift(bandwidth=30)  
    labels = mean_shift.fit_predict(intersections1)
    centroids = mean_shift.cluster_centers_

    # plt.figure(figsize=(8, 8))
    unique_clusters = set(labels)

    for cluster in unique_clusters:
        if cluster == -1: 
            continue
        mask = labels == cluster
        cluster_points = intersections1[mask]  
        color = np.random.rand(3,)  
        # plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f"Cluster {cluster}")


    for centroid in centroids:
        x, y = int(centroid[0]), int(centroid[1])
        intersections1_clustered = cv2.drawMarker(
            hough_classified1,
            (x, y),
            color=(255, 255, 255),  
            markerType=cv2.MARKER_TILTED_CROSS, 
            markerSize=10, 
            thickness=2
        )

    return centroids