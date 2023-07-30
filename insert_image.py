import cv2
import numpy as np
import time
import glob
from tensorflow.keras.applications.resnet50 import ResNet50
import json
from utils.vectorize_image import image_to_embs
from utils.milvus_ops import is_collection_exists, remove_collection, load_collection
from pymilvus import DataType, FieldSchema


def create_collection(collection_name, image_dimensions, exist_drop=True):
    is_coll_exist = is_collection_exists(collection_name)

    # drop collection is exists
    if(is_coll_exist and exist_drop):
        print(fmt.format(f"drop collection `{collection_name}`"))
        remove_collection(collection_name)

    if(exist_drop == False):
        # Create a Milvus collection to store the vectors with timestamp as index
        fields=[
            FieldSchema('empId', DataType.VARCHAR, is_primary=True, max_length=20),
            FieldSchema('empName', DataType.VARCHAR, max_length=50),
            FieldSchema('embedding', DataType.FLOAT_VECTOR, dim=image_dimensions),
            FieldSchema('timestamp', DataType.INT64)
        ]
        print(fmt.format(f"Create collection `{collection_name}`"))
        collection = create_collection(fields, collection_name)
    return collection

### Register Users
def register_users(users_config, collection_name):
    if(is_collection_exists(collection_name)):
        collection = load_collection(collection_name)
    else:
        collection = create_collection(collection_name, image_dimensions)
    reg_img_dir = "data/register"
    for user_config in users_config:
        print(f"> inserting {user_config} image")
        image_file = f"{reg_img_dir}/{user_config['emp_id']}.png"
        image = cv2.imread(image_file)
        face_image = extract_face(image)
        cv2.imwrite("face.jpg", face_image)
        image_vector = image_to_embs(face_image)

        # Insert the vector into the collection with timestamp as index
        timestamp = int(time.time())

        entities = [
            [user_config['empId']],
            [user_config['empName']],
            [image_vector],
            [timestamp],
        ]

        ### Insert records into Table
        collection.insert(entities)
    return collection

### Verify Users
def verify_user(test_dir_path, collection_name):
    collection = load_collection(collection_name)
    test_img_files = glob.glob(f"{test_dir_path}/*.png")
    for test_img_file in test_img_files:
        print(f"\n> verifying {test_img_file} image!")
        image = cv2.imread(test_img_file)
        face_image = extract_face(image)
        try:
            cv2.imwrite("face.jpg", face_image)
        except:
            print(">> face didn't recognised!")
            continue
        image_vector = image_to_embs(face_image)
        print("image_vector: ", len(image_vector))

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
                data = [image_vector],
                param = search_params,
                anns_field = 'embedding',
                limit = 2
            )
        print(">> results: ", results)



fmt = "\n=== {:30} ===\n"
# Connect to Milvus database
connections.connect(host='localhost', port='19530')
# Load pre-trained face recognition model (e.g., using OpenCV's pre-trained model)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_dimensions = 2048
MILVUS_COLLECTION_NAME = "image_collection"
threshold = 0.5

### Register Users
## Load config gile
# with open("config.json", "r") as f:
#     users_config = json.load(f)
# print("user_config: ", users_config)
# register_users(users_config, MILVUS_COLLECTION_NAME)

### Verify Users
vrf_img_dir = "data/verify"
verify_user(vrf_img_dir, MILVUS_COLLECTION_NAME)






