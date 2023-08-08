import dlib
import cv2
import numpy as np
import dlib
from utils.facealigner import FaceAligner
from config import SHAPE_PREDICTOR_MODEL_LOC
from utils.face_utils import crop_face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL_LOC)
fa = FaceAligner(predictor, desiredFaceWidth=126)


def align_face(image, bb):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 2)
    x, y, w, h = bb.astype("int")
    startX, startY, endX, endY = int(x), int(y), int(x+w), int(y+h)
    r = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))

    faceAligned, rect = fa.align(image, gray, r)
    # Update the rect with the aligned face size and position
    rect = (x, y, faceAligned.shape[1], faceAligned.shape[0])

    # Convert the aligned face back to a BGR image (for visualization)
    aligned_face_bgr = cv2.cvtColor(faceAligned, cv2.COLOR_GRAY2BGR)

    # Replace the aligned face in the original image with the transformed face
    image[y:y + faceAligned.shape[0], x:x + faceAligned.shape[1]] = aligned_face_bgr

    # Detect facial landmarks (eye points)
    landmarks = predictor(gray, rect)
    # Extract specific points for cropping
    forehead_point_1 = (landmarks.part(19).x, landmarks.part(19).y)  # Point 20 (0-based index)
    forehead_point_2 = (landmarks.part(24).x, landmarks.part(24).y)  # Point 25 (0-based index)
    left_edge_point = (landmarks.part(0).x, landmarks.part(0).y)     # Point 1
    right_edge_point = (landmarks.part(16).x, landmarks.part(16).y)   # Point 17
    bottom_edge_point = (landmarks.part(8).x, landmarks.part(8).y)    # Point 9 (jaw point)

    # Determine the top edge for cropping (use the higher of the two forehead points)
    top_edge_point = forehead_point_1 if forehead_point_1[1] < forehead_point_2[1] else forehead_point_2
    acc_crop_face = crop_face(image, top_edge_point, left_edge_point, right_edge_point, bottom_edge_point)
    return faceAligned, acc_crop_face