from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame

# --------------------------------
# Initialize pygame mixer
# --------------------------------
pygame.mixer.init()

# --------------------------------
# Load alarm sound
# --------------------------------
alarm_sound = pygame.mixer.Sound("alarm.wav.mp3")

# --------------------------------
# Eye Aspect Ratio Function
# --------------------------------
def eye_aspect_ratio(eye):

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

# --------------------------------
# Threshold Settings
# --------------------------------
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 15

COUNTER = 0
ALARM_ON = False

# --------------------------------
# Dlib Face Detector
# --------------------------------
detect = dlib.get_frontal_face_detector()

predict = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

# --------------------------------
# Eye Landmark Indexes
# --------------------------------
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# --------------------------------
# Start Webcam
# --------------------------------
cap = cv2.VideoCapture(0)

# --------------------------------
# Main Loop
# --------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)

    for subject in subjects:

        # Face Rectangle
        x = subject.left()
        y = subject.top()

        w = subject.right() - x
        h = subject.bottom() - y

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Facial Landmarks
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Eye Hulls
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(
            frame,
            [leftEyeHull],
            -1,
            (0, 255, 0),
            1
        )

        cv2.drawContours(
            frame,
            [rightEyeHull],
            -1,
            (0, 255, 0),
            1
        )

        # EAR Calculation
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Display EAR
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (450, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Sleep Detection
        if ear < EYE_AR_THRESH:

            COUNTER += 1

            cv2.putText(
                frame,
                "DROWSINESS DETECTED!",
                (130, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                cv2.putText(
                    frame,
                    "WAKE UP!",
                    (200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

                if not ALARM_ON:

                    ALARM_ON = True

                    alarm_sound.play(-1)

        else:

            COUNTER = 0

            if ALARM_ON:

                alarm_sound.stop()

            ALARM_ON = False

    # Show Window
    cv2.imshow("Sleep Detection Alarm System", frame)

    key = cv2.waitKey(1)

    # ESC key to exit
    if key == 27:
        break

# --------------------------------
# Cleanup
# --------------------------------
cap.release()
cv2.destroyAllWindows()