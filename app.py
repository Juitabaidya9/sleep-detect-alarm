import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Sleep Detection Alarm System")

st.title("😴 Sleep Detection Alarm System")
st.write("Real-time AI drowsiness detection using OpenCV and Dlib")

# ---------------------------
# EAR Function
# ---------------------------
def eye_aspect_ratio(eye):

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

# ---------------------------
# Load Detector
# ---------------------------
detect = dlib.get_frontal_face_detector()

predict = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ---------------------------
# Video Transformer
# ---------------------------
class VideoTransformer(VideoTransformerBase):

    COUNTER = 0

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        subjects = detect(gray, 0)

        for subject in subjects:

            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            cv2.putText(
                img,
                f"EAR: {ear:.2f}",
                (400, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if ear < 0.23:

                self.COUNTER += 1

                cv2.putText(
                    img,
                    "DROWSINESS DETECTED!",
                    (120, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3
                )

            else:
                self.COUNTER = 0

        return img

# ---------------------------
# Start Webcam
# ---------------------------
webrtc_streamer(
    key="sleep-detection",
    video_transformer_factory=VideoTransformer,
)