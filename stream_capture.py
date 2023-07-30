from imutils.video import FileVideoStream
from imutils.video import VideoStream
import time
import argparse
import imutils
import cv2
from utils.detect_eye_blinks import DetectEyes

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True, help='path to facial landmark predictor')
ap.add_argument('-v', '--video', type=str, default="", help='path to input video file')
args = vars(ap.parse_args())

print('[INFO] Starting video stream thread...')
fileStream = False
if args['video']:
    vs = FileVideoStream(args['video']).start()
    fileStream = True
else:
    vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start()
    fileStream = False
time.sleep(1.0)

eyeDet = DetectEyes(shape_predictor_path=args['shape_predictor'])

while True:
    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    orig_frame = frame.copy()
    processed_frame = eyeDet.detect_eye_blinks(frame)
    if(eyeDet.TOTAL_BLINKS == 3):
        print("[INFO] User Eye Blinks Verified!")
        cv2.imwrite("user_image.jpg", orig_frame)
        break

    cv2.imshow("Frame", processed_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
