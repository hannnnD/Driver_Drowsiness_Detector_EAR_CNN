import cv2
import os
import dlib
import pygame
import numpy as np
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model

# Initialize Pygame for playing sound
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), 'audio', 'alert.wav'))

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

# Load the trained CNN model
cnn_model = load_model(os.path.join(os.path.dirname(__file__), 'drowsiness_detection_cnn_model_v35.h5'))

# EAR calculation function
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (1.2 * A + B) / (2.0 * C)
    return ear

# Indices for the left and right eyes
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# EAR threshold and frame counters
EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 20
score = 0

# Function to preprocess image for CNN model
def preprocess_image(face_roi):
    face_resized = cv2.resize(face_roi, (64, 64))
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=-1)  # Add channel dimension
    face_expanded = np.expand_dims(face_expanded, axis=0)    # Add batch dimension
    return face_expanded

# Main loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        score += 1
        cv2.putText(frame, "No face detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_roi = gray[y:y+h, x:x+w]

            # Preprocess for CNN model
            processed_face = preprocess_image(face_roi)
            cnn_prediction = cnn_model.predict(processed_face)
            cnn_label = np.argmax(cnn_prediction)

            # Detect facial landmarks
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # Calculate EAR for both eyes
            left_eye = shape[LEFT_EYE]
            right_eye = shape[RIGHT_EYE]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Display CNN prediction
            state = "Awake" if cnn_label == 0 else "Drowsy"
            cv2.putText(frame, f"CNN: {state}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Combine EAR and CNN predictions
            if ear < EAR_THRESHOLD or cnn_label == 1:
                score += 1
                cv2.putText(frame, "Eyes Closed", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                score = max(0, score - 1)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the score on the top-right corner of the screen
    cv2.putText(frame, f"Score: {score}", (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Trigger alarm if score exceeds threshold
    if score > CONSEC_FRAMES:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        pygame.mixer.music.stop()

    # Display the resulting frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop on 'q' or 'ESC' key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:  # 27 is the ASCII code for ESC
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
