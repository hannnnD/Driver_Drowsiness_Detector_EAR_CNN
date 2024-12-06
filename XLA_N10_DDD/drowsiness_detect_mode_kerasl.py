# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame
import time
import dlib
import cv2
import os
from tensorflow.keras.models import load_model

# Initialize Pygame and load alert sounds
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), 'audio', 'alert.wav'))
no_face_sound = pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), 'audio', 'No_Face.wav'))

# Threshold for eye aspect ratio and number of consecutive frames for alarm
EYE_ASPECT_RATIO_THRESHOLD = 0.3
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
COUNTER = 0

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

# Load the pre-trained Keras model for classification
model = load_model(os.path.join(os.path.dirname(__file__), 'ddm_v1.h5'))

# Extract the indexes of facial landmarks for the eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using dlib
    faces = detector(gray, 0)

    # If no faces are detected, play a sound
    if len(faces) == 0:
        cv2.putText(frame, "No face detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        pygame.mixer.Sound.play(no_face_sound)
    else:
        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Get coordinates for the eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Calculate eye aspect ratio (EAR) for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # If EAR is below the threshold, increment the frame counter
            if ear < EYE_ASPECT_RATIO_THRESHOLD:
                COUNTER += 1

                # If the eyes are closed for a sufficient number of frames, play the alarm
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()

                    # Display "Drowsiness Alert" on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                pygame.mixer.music.stop()

            # Preprocess the face region for classification (using Keras model)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)

            # Predict drowsiness using the model
            preds = model.predict(face_roi)[0]
            drowsy_class = np.argmax(preds)
            confidence = preds[drowsy_class] * 100

            # Set label and color for different states
            if drowsy_class == 0 or drowsy_class == 3:  # Drowsy states
                label = "Drowsy"
                color = (0, 0, 255)  # Red
            elif confidence < 60:  # Slightly drowsy or unsure state
                label = "Slightly Drowsy"
                color = (0, 255, 255)  # Yellow
            else:  # Alert state
                label = "Alert"
                color = (0, 255, 0)  # Green

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Display the prediction label and confidence score
            cv2.putText(frame, f"State: {label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}%", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the resulting frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop on 'q' or 'ESC' key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:  # 27 is the ASCII code for ESC
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
