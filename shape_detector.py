import cv2
import dlib
import numpy as np

img = cv2.imread('data/register/images/Rohan/Rohan-5.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 1) # rects contains all the faces detected

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

predictor = dlib.shape_predictor('data/models_artifactory/shape_predictor_68_face_landmarks.dat')
for (i, rect) in enumerate(rects):
    print(rect, type(rect))
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    for (x, y) in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    cv2.imwrite("shape_points.png", img)