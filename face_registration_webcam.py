import cv2
import time
from imutils.video import WebcamVideoStream, VideoStream
from utils.milvus_ops import MilvusOps
from utils.vectorize_image import faceimg_to_embs
from utils.detect_faces import extract_faces
from config import FACE_DETECTION_MODEL_TYPE, MILVUS_COLLECTION_NAME, REG_VIDEOSTREAM_LOC

def save_details_in_milvus(user_info, img_frame, milvus_ops, model_type):
    # Insert the vector into the collection with timestamp as index
    timestamp = int(time.time())
    cropped_img, is_face_found, _ = extract_faces(img_frame, model_type=model_type)
    if(is_face_found):
        face_vector = faceimg_to_embs(cropped_img, multi_faces=False)
        entities = [
            [user_info['empId']],
            [user_info['empName']],
            [face_vector],
            [timestamp],
        ]

        ### Insert records into Table
        return milvus_ops.add_to_collection(entities)
    return None

def register_process():
    ### Stage-1: Take input from user
    input_stage = True
    emp_id = None
    while(input_stage):
        emp_name = str(input("Enter Your Name: ")).strip()
        emp_id = str(input("Enter Employee ID: ")).strip()
        print(f"Verify Entered Details:\nName: {emp_name}\nEmployeeId: {emp_id}\n")
        print("-"*10)
        while(True):
            move_ahead = str(input("Proceed ahead(y/Y) or re-input details(n/N): ")).strip()
            if(move_ahead in ['y', 'Y']):
                input_stage = False
                break
            elif(move_ahead in ['n', 'N']):
                break
            else:
                print("wrong option! re-select options...")
                pass


    ### Stage-2: Capture Pictures & Save
    print(">> Setting up Milvus Connection...")
    milvus_ops = MilvusOps(MILVUS_COLLECTION_NAME)
    print("<< Milvus Connection Created!")
    user_info = {}
    user_info['empName'] = emp_name
    user_info['empId'] = emp_id

    print("Be Ready...(Will capture your pictures)")
    if(REG_VIDEOSTREAM_LOC == 'webcam'):
        vs = WebcamVideoStream(src=0).start()
    else:
        vs = VideoStream(src=REG_VIDEOSTREAM_LOC).start()

    print("> Video/Camera Processing on...")
    time.sleep(3) # To avoid initial blurred frames
    while(True):
        frame = vs.read()
        frame = cv2.flip(frame, 1) # Flip to act as a mirror
        cv2.imshow("Video Stream: ", frame)
        key = cv2.waitKey(1) & 0xFF
        if(key == ord('q')):
            print("Quitting...")
            break
        elif(key == ord('s')):
            print("Saving the frame in Milvus...")
            # Destroy all the windows
            vs.stop()
            cv2.destroyAllWindows()

            response = save_details_in_milvus(user_info, frame, milvus_ops, model_type=FACE_DETECTION_MODEL_TYPE)
            if(response == None):
                continue
            print(f"response: {response}")
            break

    print("Operations are done!")
    print('-'*10)
    milvus_ops.close_connection() ### Release Milvus Resources


### Start the process
register_process()
