import cv2
import os
import numpy as np
import dlib
from tensorflow.keras.models import load_model
import pygame

# Initialize Pygame for playing sound
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), 'audio', 'alert.wav'))

# Load pre-trained Keras model for drowsiness detection
model = load_model(os.path.join(os.path.dirname(__file__), 'drowsiness_detection_model.h5'))

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using dlib
    faces = detector(gray)

    if len(faces) == 0:
        # Display "No face detected" message
        cv2.putText(frame, "No face detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        pygame.mixer.music.stop()
    else:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Extract the region of interest (ROI) for the face
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.shape[0] <= 0 or face_roi.shape[1] <= 0:
                continue  # Skip if face ROI is invalid

            # Preprocess the face for Keras model
            face_roi_resized = cv2.resize(face_roi, (64, 64))
            face_roi_normalized = face_roi_resized / 255.0
            face_roi_reshaped = np.expand_dims(face_roi_normalized, axis=0)

            # Predict drowsiness using the Keras model
            preds = model.predict(face_roi_reshaped)[0]
            drowsy_class = np.argmax(preds)
            confidence = preds[drowsy_class] * 100

            # Set label and color based on prediction
            if drowsy_class == 0:  # Drowsy
                label = "Drowsy"
                color = (0, 0, 255)  # Red
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:  # Alert
                label = "Alert"
                color = (0, 255, 0)  # Green
                pygame.mixer.music.stop()

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Display the prediction label and confidence score
            cv2.putText(frame, f"{label} ({confidence:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
