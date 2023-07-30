from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from config import SHAPE_PREDICTOR_MODEL_LOC
import dlib
import cv2
from utils.face_utils import crop_face
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_MODEL_LOC)

# Define a dataset and data loader
def dataset_loader(dir_path):
    dataset = datasets.ImageFolder(dir_path)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=lambda x: x[0])
    return dataset, loader


def face_post_process(image, rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    return acc_crop_face