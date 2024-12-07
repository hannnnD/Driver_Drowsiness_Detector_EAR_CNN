import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
import dlib
import seaborn as sns  # Để vẽ confusion matrix dễ dàng hơn

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

# Load the CNN model
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

# Preprocess image for CNN
def preprocess_image(face_roi):
    face_resized = cv2.resize(face_roi, (64, 64))
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=-1)  # Add channel dimension
    face_expanded = np.expand_dims(face_expanded, axis=0)    # Add batch dimension
    return face_expanded

# Function to process an image
def process_image(image_path, ear_threshold=0.3):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return -1  # Nhãn 'unknown'

    faces = detector(image)
    if len(faces) == 0:
        return -1  # Nhãn 'unknown' nếu không phát hiện được khuôn mặt

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Đảm bảo tọa độ bounding box hợp lệ
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)

        face_roi = image[y:y+h, x:x+w]

        # Bỏ qua khuôn mặt nếu ROI không hợp lệ
        if face_roi.size == 0:
            continue

        # Tính EAR
        shape = predictor(image, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        left_eye = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Dự đoán CNN
        processed_face = preprocess_image(face_roi)
        cnn_prediction = cnn_model.predict(processed_face)
        cnn_label = np.argmax(cnn_prediction)

        # Kết hợp EAR và CNN
        if ear < ear_threshold or cnn_label == 1:
            return 1  # Drowsy
        else:
            return 0  # Non-drowsy

    return -1  # Nhãn 'unknown' nếu không có ROI hợp lệ

# Evaluate on the dataset
def evaluate_model(dataset_path):
    y_true = []
    y_pred = []
    total_valid_samples = 0  # Đếm số lượng ảnh được xử lý (không nhãn -1)

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        class_label = 1 if label == "Drowsy" else 0
        
        # Lấy danh sách tối đa 1000 ảnh từ thư mục con
        image_names = os.listdir(label_path)[:1000]
        
        for image_name in image_names:
            image_path = os.path.join(label_path, image_name)
            predicted_label = process_image(image_path)

            if predicted_label == -1:  # Bỏ qua các ảnh không nhận diện được
                continue

            total_valid_samples += 1  # Chỉ tăng với ảnh xử lý thành công
            y_true.append(class_label)
            y_pred.append(predicted_label)

    # Kiểm tra nếu có dữ liệu để đánh giá
    if total_valid_samples == 0:
        raise ValueError("No valid samples processed for evaluation.")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, conf_matrix,  y_true, y_pred

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-drowsy", "Drowsy"], yticklabels=["Non-drowsy", "Drowsy"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Run evaluation
dataset_path = r"D:\Download\Driver-Drowsiness-Detector-master\XLA_N10_DDD\Dataset_DriverDrowsines(DDD)"
accuracy, precision, recall, f1, conf_matrix, y_true, y_pred = evaluate_model(dataset_path)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
class_report = classification_report(
    y_true, 
    y_pred, 
    target_names=["Non-drowsy", "Drowsy"]
)
print("\n=== Detailed Classification Report ===")
print(class_report)
# Plot confusion matrix
plot_confusion_matrix(conf_matrix)
