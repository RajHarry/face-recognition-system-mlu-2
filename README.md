## data folder structure:
- data/
    - models_artifactory/
        - haarcascade_eye.xml
        - haarcascade_frontalface_default.xml
        - haarcascade_mcs_nose.xml
        - shape_predictor_68_face_landmarks.dat
    - processed/
        - register/
        - verify/
    - raw-videos/
    - register/
        - images/
            - user-1/
                - user-1-1.png
                - user-1-2.png
            - user-2/
                - user-2-1.png
                - user-2-2.png
        - reg_config.json
    - verify/
        - Images/


## Models Setup:

* Download "haarcascade_frontalface_default.xml" from https://github.com/opencv/opencv/tree/master/data/haarcascades
* Download "shape_predictor_68_face_landmarks.dat" from https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

* Directory to put: data/models_artifactory


## Planned Improvements:
* Approach-1: Multiple faces store per user
    * While Registration:
        * take multiple faces of the user
        * Extract embeddings for every face (ex: 512 embedding size)
        * combine them as one embedding
            * 1536 embedding size (512*3 -> 512 embedding size for each)
    * While Verification:
        * take a single face
        * extract embeddings of that face (ex: 512 embedding size)
        * combine same embeddings thrice([0-511, 512-1024, 1024-1536])
        * search for vector in milvus db
* Approach-2: yolov7 face detection(https://github.com/elyha7/yoloface)