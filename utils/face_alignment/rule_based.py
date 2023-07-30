### Reference
# https://www.geeksforgeeks.org/face-alignment-with-opencv-and-python/

# install and import above modules first
import cv2
import math
from PIL import Image
import numpy as np
from utils.face_utils import euclidean_distance
from config import FACE_DETECTOR_CASCADE_MODEL, EYE_DETECTOR_CASCADE_MODEL

FACE_DETECTOR = cv2.CascadeClassifier(FACE_DETECTOR_CASCADE_MODEL)
EYE_DETECTOR = cv2.CascadeClassifier(EYE_DETECTOR_CASCADE_MODEL)

# Detect face
def face_detection(img):
    faces = FACE_DETECTOR.detectMultiScale(img, 1.1, 4)
    if(len(faces) == 1):
        X, Y, W, H = faces[0]
        img = img[int(Y):int(Y+H), int(X):int(X+W)]
        return img, cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        return None, None

# Find eyes
def align_face(img_raw):
    img, gray_img = face_detection(img_raw)
    if(img is None):
        return img_raw
    eyes = EYE_DETECTOR.detectMultiScale(gray_img)

    # deciding to choose left and right eye
    eye_1 = eyes[0]
    eye_2 = eyes[1]
    if eye_1[0] > eye_2[0]:
        left_eye = eye_2
        right_eye = eye_1
    else:
        left_eye = eye_1
        right_eye = eye_2

    # center of eyes
    # center of right eye
    right_eye_center = (
        int(right_eye[0] + (right_eye[2]/2)),
        int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]
    right_eye_y = right_eye_center[1]
    # cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)

    # center of left eye
    left_eye_center = (
        int(left_eye[0] + (left_eye[2] / 2)),
        int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]
    left_eye_y = left_eye_center[1]
    # cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)

    # finding rotation direction
    if left_eye_y > right_eye_y:
        print("Rotate image to clock direction")
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate image direction to clock
    else:
        print("Rotate to inverse clock direction")
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, point_3rd)
    c = euclidean_distance(right_eye_center, left_eye_center)
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = (np.arccos(cos_a) * 180) / math.pi

    if direction == -1:
        angle = 90 - angle
    else:
        angle = -(90-angle)

    # rotate image
    new_img = Image.fromarray(img_raw)
    new_img = np.array(new_img.rotate(direction * angle))

    return new_img

# image = cv2.imread("data/register/images/Rohan/Rohan-5.png")
# alignedFace = align_face(image)