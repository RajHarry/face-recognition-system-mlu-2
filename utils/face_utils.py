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