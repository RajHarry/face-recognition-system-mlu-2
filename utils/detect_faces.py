import cv2
from facenet_pytorch import MTCNN
import torch
from utils.data_process import dataset_loader
from config import FACE_DETECTOR_CASCADE_MODEL, DEVICE

# Load pre-trained face recognition model (e.g., using OpenCV's pre-trained model)
face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_CASCADE_MODEL)

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    device=DEVICE,
    keep_all=True
)
# ### For full resolution detections
# fast_mtcnn = FastMTCNN(
#     stride=4,
#     resize=1,
#     margin=14,
#     factor=0.6,
#     keep_all=True,
#     device=device
# )



def extract_faces_cascade(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(64, 64))
                                            # gray,scaleFactor=1.2,minNeighbors=10,minSize=(64,64),flags=cv2.CASCADE_SCALE_IMAGE)
    no_of_faces = len(face_boxes)
    is_face_found = False
    cropped_face = []
    if(no_of_faces == 1):
        is_face_found = True
        (x, y, w, h) = face_boxes[0]
        cropped_face = image[y:y+h, x:x+w]
        return cropped_face, is_face_found, face_boxes
    else:
        return [], is_face_found, []

def extract_faces_mtcnn(image):
    # Get cropped and prewhitened image tensor
    face_boxes = []
    cropped_face = mtcnn(image)
    is_face_found = False
    if(cropped_face is not None):
        face_boxes, prob = mtcnn.detect(image)
        if(face_boxes is not None):
            is_face_found = True
            face_boxes = list(face_boxes)

    return cropped_face, is_face_found, face_boxes

def extract_faces_mtcnn_v2(image):
    # Get cropped and prewhitened image tensor
    face_boxes = []
    face_boxes, prob = mtcnn.detect(image)
    is_face_found = False
    if(len(face_boxes)==1):
        is_face_found = True
        face_boxes = list(face_boxes)
        (x, y, w, h) = face_boxes[0]
        cropped_face = image[y:y+h, x:x+w]
        return cropped_face, is_face_found, face_boxes
    else:
        return [], is_face_found, []

def extract_faces_fastmtcnn(image):
    # Get cropped and prewhitened image tensor
    # face_boxes = []
    cropped_face, face_boxes = fast_mtcnn(image)
    # save_location = f"{ROOT_DIR}/{save_location}"
    # # img = Image.fromarray(cropped_face.cpu().detach().numpy()[0])
    # # img.save(save_location)
    is_face_found = False
    if(cropped_face is not None):
        # face_boxes, prob = mtcnn.detect(image)
        if(face_boxes is not None):
            is_face_found = True
            face_boxes = list(face_boxes)

    return cropped_face, is_face_found, face_boxes

def extract_faces(img_frame, model_type):
    if(model_type == 'cascade'):
        cropped_img, is_face_found, face_boxes = extract_faces_cascade(img_frame)

    elif(model_type == 'mtcnn'):
        # cropped_img, is_face_found, face_boxes = extract_faces_mtcnn(img_frame)
        cropped_img, is_face_found, face_boxes = extract_faces_mtcnn_v2(img_frame)

    elif(model_type == 'fastmtcnn'):
        cropped_img, is_face_found, face_boxes = extract_faces_fastmtcnn(img_frame)

    return cropped_img, is_face_found, face_boxes

# Perfom MTCNN facial detection
def mtcnn_facial_detection(dir_path):
    # Define a dataset and data loader
    dataset, loader = dataset_loader(dir_path)

    #Perfom MTCNN facial detection
    #Detects the face present in the image and prints the probablity of face detected in the image.
    aligned = []
    names = []
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    # Calculate the 512 face embeddings
    aligned = torch.stack(aligned).to(DEVICE)
    embeddings = resnet(aligned).cpu()

    return embeddings, names