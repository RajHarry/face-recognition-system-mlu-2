from facenet_pytorch import InceptionResnetV1
import torch
import torchvision.transforms as transforms
from config import PROCESS_RUN_ON, EMBEDDING_MODEL, DEVICE

# Define Inception Resnet V1 module (GoogLe Net)
resnet = InceptionResnetV1(pretrained=EMBEDDING_MODEL).eval().to(DEVICE)
transform_to_tensor = transforms.ToTensor()

# Embed the face-image as a vector using a pre-trained model
def faceimg_to_embs(cropped_face, multi_faces=False):
    cropped_face = transform_to_tensor(cropped_face) # transform cv2 numpy array to tensor
    if(multi_faces):
        embeddings = resnet(cropped_face.unsqueeze(0)).cpu()
    else:
        embeddings = resnet(cropped_face.unsqueeze(0))
    return embeddings.tolist()[0]