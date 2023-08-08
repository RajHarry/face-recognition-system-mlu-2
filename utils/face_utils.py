import math
import numpy as np

def euclidean_distance(a, b):
    x1, x2 = a[0], b[0]
    y1, y2 = a[1], b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def bb_to_rect(bb):
    top=bb[1]
    left=bb[0]
    right=bb[0]+bb[2]
    bottom=bb[1]+bb[3]
    return np.array([top, right, bottom, left])

def crop_face(image, top_edge_point, left_edge_point, right_edge_point, bottom_edge_point):
    # Define the new bounding box for the face crop
    top_left = (left_edge_point[0], top_edge_point[1])
    bottom_right = (right_edge_point[0], bottom_edge_point[1])

    # Crop the face region
    cropped_face = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return cropped_face