import os
import cv2
import dlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến dataset
dataset_dir = r"d:\Download\Driver-Drowsiness-Detector-master\XLA_N10_DDD\Dataset_DriverDrowsines(DDD)"

# Khởi tạo dlib face detector và facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))

# Hàm tính EAR
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Khoảng cách dọc
    B = np.linalg.norm(eye[2] - eye[4])  # Khoảng cách dọc
    C = np.linalg.norm(eye[0] - eye[3])  # Khoảng cách ngang
    ear = (1.2 * A + B) / (2.0 * C)
    return ear

# Indices của mắt trái và mắt phải trong landmark
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Hàm tiền xử lý và tính EAR từ ảnh
def preprocess_and_calculate_ear(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)  # Áp dụng GaussianBlur
    faces = detector(gray)
    if len(faces) == 0:
        return None  # Không tìm thấy khuôn mặt

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        return (left_ear + right_ear) / 2.0  # EAR trung bình cho cả hai mắt
    return None

# Load dữ liệu EAR từ dataset
def load_ear_dataset(dataset_dir, max_images_per_class=1000):
    ear_data = []
    labels = []
    for label, class_name in enumerate(["Non_Drowsy", "Drowsy"]):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        images = os.listdir(class_dir)[:max_images_per_class]  # Lấy tối đa max_images_per_class ảnh
        for image_name in images:
            image_path = os.path.join(class_dir, image_name)
            try:
                ear = preprocess_and_calculate_ear(image_path)
                if ear is not None:
                    ear_data.append(ear)
                    labels.append(label)
            except Exception as e:
                print(f"Lỗi xử lý ảnh {image_path}: {e}")

    # Chuyển thành numpy array
    X = np.array(ear_data).reshape(-1, 1)  # EAR là đặc trưng
    y = np.array(labels)  # Nhãn
    return X, y

# Load dữ liệu EAR từ dataset
X, y = load_ear_dataset(dataset_dir)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)

# Hiển thị các kết quả đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Trực quan hóa Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Non_Drowsy", "Drowsy"], yticklabels=["Non_Drowsy", "Drowsy"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
