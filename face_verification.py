import cv2
from imutils.video import WebcamVideoStream, VideoStream, FileVideoStream
import time
from datetime import datetime
from utils.milvus_ops import MilvusOps
from config import FACE_DETECTION_MODEL_TYPE, MILVUS_COLLECTION_NAME, MILVUS_DISTANCE_THRESHOLD, \
                VERIFY_VIDEOSTREAM_LOC, USE_ALIGNMENT, PROCESSED_IMG_VERIFY_SAVE_LOC
from utils.vectorize_image import faceimg_to_embs
from utils.detect_faces import extract_faces
from utils.face_align import align_face
import json
from utils.data_process import face_post_process
import dlib

### Vector searching of employees
def user_search(img_frame, milvus_ops, model_type):
    cropped_img, is_face_found, face_boxes = extract_faces(img_frame, model_type=model_type)
    if(is_face_found):
        cv2.imwrite(f"{PROCESSED_IMG_VERIFY_SAVE_LOC}/face_detected_{model_type}.jpg", cropped_img)
        face_box = face_boxes[0] # Multiple faces can also be found
        if(USE_ALIGNMENT):
            cropped_img = align_face(img_frame, face_box)

        x, y, w, h = face_box
        startX, startY, endX, endY = int(x), int(y), int(x+w), int(y+h)
        rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
        acc_crop_face = face_post_process(img_frame, rect)
        cv2.imwrite(f"{PROCESSED_IMG_VERIFY_SAVE_LOC}/face_acc_cropped_{model_type}.jpg", acc_crop_face)
        face_vector = faceimg_to_embs(acc_crop_face, multi_faces=False)

        ### User Vector Search
        response = milvus_ops.vector_search(face_vector)
        print("response: ", response)
        result = response[0]
        distance = result.distances[0]
        if(distance < MILVUS_DISTANCE_THRESHOLD):
            extracted_details = str(result).split("entity:")[-1].replace("'", "\"")[:-2]
            entities = json.loads(extracted_details)
            entities['distance'] = "{:.2f}".format(distance)
            return entities, face_box
    return None, None

def run_process():
    ### Stage-1: Capture Pictures & search
    print(">> Setting up Milvus Connection...")
    milvus_ops = MilvusOps(MILVUS_COLLECTION_NAME)
    print("<< Milvus Connection Created!")

    print("Be Ready...(Will capture your pictures)")
    is_webcam = False
    if(VERIFY_VIDEOSTREAM_LOC == 'webcam'):
        is_webcam = True
        vs = WebcamVideoStream(src=0).start()
    else:
        vs = FileVideoStream(VERIFY_VIDEOSTREAM_LOC).start()

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)
    thickness = 2 #(try differnt values)
    fontScale = 1 #(try differnt values)
    print("> Video/Camera Processing on...")
    time.sleep(3) # To avoid initial blurred frames
    while(vs.more()):
        t1 = datetime.now()
        frame = vs.read()
        if(frame is None):
            print(">> frame is None...")
            pass
        frame = cv2.flip(frame, 1) # Flip to act as a mirror
        response, face_box = user_search(frame, milvus_ops, model_type=FACE_DETECTION_MODEL_TYPE) # model_type = ['cascade', 'mtcnn']
        if(response is not None or face_box is not None):
            (x, y, w, h) = face_box
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            org = (x, y)
            label = f"{response['empName']}__{response['distance']}"
            cv2.putText(frame, label, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
            print(f"response: {response} in {datetime.now()-t1}S")
        cv2.imshow("Video Stream: ", frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            print("Quitting...")
            break

    print("Operations are done!")
    print('-'*10)

    # Destroy all the windows
    vs.stop()
    cv2.destroyAllWindows()
    milvus_ops.close_connection()

run_process()
