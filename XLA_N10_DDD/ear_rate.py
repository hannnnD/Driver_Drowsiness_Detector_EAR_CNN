import os
import cv2
import dlib
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

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

# Apply Gaussian Blur
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """Apply Gaussian Blur to reduce noise."""
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

# Dataset paths
dataset_path = r"d:\Download\Driver-Drowsiness-Detector-master\XLA_N10_DDD\Dataset_DriverDrowsines(DDD)"
categories = ["Non Drowsy", "Drowsy"]
ear_data = []

# Gaussian parameters
kernel_sizes = [(3, 3), (5, 5), (7, 7)]
sigma_values = [0, 1, 2]

# Process images
for category in categories:
    print(f"Processing category: {category}")
    label = 0 if category == "Non Drowsy" else 1
    category_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        print(f"Processing image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image {img_path} could not be read. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Test Gaussian parameters and find the best combination
        best_ear = None
        for kernel_size in kernel_sizes:
            for sigma in sigma_values:
                blurred = apply_gaussian_blur(gray, kernel_size, sigma)
                faces = detector(blurred)
                
                if len(faces) > 0:
                    for face in faces:
                        shape = predictor(blurred, face)
                        shape = np.array([[p.x, p.y] for p in shape.parts()])

                        # Calculate EAR
                        left_eye = shape[LEFT_EYE]
                        right_eye = shape[RIGHT_EYE]
                        left_ear = calculate_ear(left_eye)
                        right_ear = calculate_ear(right_eye)
                        ear = (left_ear + right_ear) / 2.0

                        # Update the best EAR value
                        if best_ear is None or ear > best_ear:
                            best_ear = ear

        # Append EAR and label if a valid EAR is found
        if best_ear is not None:
            ear_data.append([best_ear, label])
        else:
            print(f"No valid EAR calculated for image: {img_path}")

    print(f"Finished processing category: {category}")

# Save to CSV
df = pd.DataFrame(ear_data, columns=["EAR", "Label"])
df.to_csv("ear_data_images_with_blur.csv", index=False)
print("EAR data saved to ear_data_images_with_blur.csv")
