import os
import torch

### General
# The model is running on CPU, since it is already pre-trained and doesnt require GPU
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(DEVICE))

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_IMG_REG_SAVE_LOC = f'{ROOT_DIR}/data/processed/register'
PROCESSED_IMG_VERIFY_SAVE_LOC = f'{ROOT_DIR}/data/processed/verify'
PROCESS_RUN_ON = 'cpu' # 'cpu' or 'cuda:0'
EMBEDDING_MODEL = 'vggface2'
MODELS_ARTIFACTORY = 'data/models_artifactory'

### Input Video Stream
REG_BY_IMAGES = True
REG_BY_IMAGES_CONFIG = 'data/register/reg_config.json'

REG_VIDEOSTREAM_LOC = 'data/raw-videos/raghava_reg_1.mov' # 'webcam' -> webcam, 'name.mp4' -> pre-recorded video
VERIFY_VIDEOSTREAM_LOC = 'data/raw-videos/Rajkumar_reg_2.mov' # 'webcam' -> webcam, 'name.mp4' -> pre-recorded video
### Face Detection
FACE_DETECTION_MODEL_TYPE_TYPE = 'cascade'
FACE_DETECTOR_CASCADE_MODEL = f'{ROOT_DIR}/{MODELS_ARTIFACTORY}/haarcascade_frontalface_default.xml'
EYE_DETECTOR_CASCADE_MODEL = f'{ROOT_DIR}/{MODELS_ARTIFACTORY}/haarcascade_eye.xml'

### Face Alignment
USE_ALIGNMENT = False
SHAPE_PREDICTOR_MODEL_LOC = f'{ROOT_DIR}/{MODELS_ARTIFACTORY}/shape_predictor_68_face_landmarks.dat'

### Milvus VectorDB
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
MILVUS_COLLECTION_NAME = "employees"
MILVUS_DISTANCE_THRESHOLD = 0.5