import cv2
import numpy as np
from milvus import Milvus, IndexType, MetricType
from pymilvus import DataType

# Milvus configuration
MILVUS_HOST = 'localhost'  # Modify this if your Milvus server is running on a different host
MILVUS_PORT = '19530'  # Modify this if your Milvus server is running on a different port
MILVUS_COLLECTION_NAME = 'face_collection'

# Initialize Milvus client
milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)



def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, (112, 112))

def add_face_to_collection(face_image):
    face = extract_face(face_image)
    if face is not None:
        # Convert face image to vector representation
        face_vector = np.array(face).flatten().tolist()

        # Add face vector to Milvus collection
        status, ids = milvus.add_vectors(collection_name=MILVUS_COLLECTION_NAME, records=[face_vector])
        if status.OK():
            print("Face added to Milvus collection successfully.")
            return ids[0]
        else:
            print("Failed to add face to Milvus collection:", status)
    else:
        print("No face found in the image.")
    return None

def verify_face(face_image):
    face = extract_face(face_image)
    if face is not None:
        # Convert face image to vector representation
        face_vector = np.array(face).flatten().tolist()

        # Search for similar faces in Milvus collection
        status, results = milvus.search_vectors(collection_name=MILVUS_COLLECTION_NAME,
                                               query_records=[face_vector],
                                               top_k=1,
                                               params={'metric_type': MetricType.L2})
        if status.OK():
            if len(results[0]) > 0 and results[0][0].distance < threshold:  # Adjust the threshold as per your requirement
                print("Face verified successfully.")
                return results[0][0].id
            else:
                print("Face verification failed.")
        else:
            print("Failed to search for similar faces:", status)
    else:
        print("No face found in the image.")
    return None

# Main program
if __name__ == '__main__':
    # Connect to Milvus server
    milvus.connect()
    threshold = 0.5

    # Create a collection in Milvus if it doesn't already exist
    if not milvus.has_collection(collection_name=MILVUS_COLLECTION_NAME):
        milvus.create_collection(collection_name=MILVUS_COLLECTION_NAME,
                                 fields=[
                                     {'name': 'vector', 'type': DataType.FLOAT_VECTOR, 'dim': 112}
                                 ],
                                 index_file_size=1024,
                                 metric_type=MetricType.L2)

    # Example usage
    image_path = 'data/trump.png'
    face_image = cv2.imread(image_path)

    # Add face to Milvus collection
    face_id = add_face_to_collection(face_image)

    # Verify face against stored faces
    verify_face(face_image)

    # Disconnect from Milvus server
    milvus.disconnect()
