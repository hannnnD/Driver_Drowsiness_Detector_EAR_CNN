import cv2
import os
import dlib
import pygame
import numpy as np
from scipy.spatial import distance as dist

# Initialize Pygame for playing sound
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), 'audio', 'alert.wav'))

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

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

# Thêm bước lọc Gaussian vào xử lý ảnh
def apply_gaussian_blur(gray, kernel_size=(5, 5), sigma=0):
    """Apply Gaussian Blur to reduce noise."""
    return cv2.GaussianBlur(gray, kernel_size, sigma)

# Cập nhật trong vòng lặp video
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_gaussian_blur(gray, kernel_size=(5, 5), sigma=1)  # Dùng kernel_size=5 và sigma=1 mặc định
    faces = detector(gray)

    if len(faces) == 0:
        # Increase score if no face detected
        score += 1
        cv2.putText(frame, "No face detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Detect facial landmarks
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # Calculate EAR for both eyes
            left_eye = shape[LEFT_EYE]
            right_eye = shape[RIGHT_EYE]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Draw boxes around eyes and display EAR values
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Check if EAR is below threshold
            if ear < EAR_THRESHOLD:
                score += 1
                cv2.putText(frame, "Eyes Closed", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                score = max(0, score - 1)  # Decrease score if eyes open

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the score on the top-right corner of the screen
    cv2.putText(frame, f"Score: {score}", (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Trigger alarm if score exceeds threshold
    if score > CONSEC_FRAMES:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 120),
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
