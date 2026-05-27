# Sleep Detection Alarm System 😴🚨

An AI-powered sleep detection system using Python, OpenCV, Dlib, and Pygame.

This project detects eye closure in real-time using a webcam and plays an alarm sound when drowsiness is detected.

---

## Features

- Real-time face detection
- Eye aspect ratio (EAR) calculation
- Drowsiness detection
- Alarm sound when eyes remain closed
- Webcam live monitoring

---

## Technologies Used

- Python
- OpenCV
- Dlib
- Scipy
- Imutils
- NumPy
- Pygame

---

## Project Structure

```bash
sleep-detect-alarm/
│
├── main.py
├── alarm.wav.mp3
├── shape_predictor_68_face_landmarks.dat
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Juitabaidya9/sleep-detect-alarm.git
cd sleep-detect-alarm
```

---

### 2. Create Virtual Environment

```bash
py -3.10 -m venv venv
```

Activate environment:

```bash
.\venv\Scripts\Activate
```

---

### 3. Install Dependencies

```bash
pip install opencv-python scipy imutils numpy pygame dlib
```

---

## Run Project

```bash
python main.py
```

---

## How It Works

- Webcam captures face
- Facial landmarks detect eye positions
- Eye Aspect Ratio (EAR) is calculated
- If eyes remain closed for several frames:
  - Drowsiness detected
  - Alarm sound plays

---

## Author

### Juita Baidya

GitHub:
https://github.com/Juitabaidya9

---

## Output

- Detects closed eyes
- Shows live webcam feed
- Plays alarm during drowsiness

---

## License

This project is for educational purposes.