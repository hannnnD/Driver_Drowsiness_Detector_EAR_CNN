import os
import random
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến mô hình và dataset
current_dir = os.path.dirname(__file__)
cnn_model_path = os.path.join(current_dir, 'drowsiness_detection_cnn_model_v35.h5')
dataset_dir = r"D:\Download\Driver-Drowsiness-Detector-master\XLA_N10_DDD\Dataset_DriverDrowsines(DDD)"

# Load mô hình CNN
model = load_model(cnn_model_path)

# Tiền xử lý ảnh
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Đọc ảnh
    if image is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang Grayscale
    image = cv2.resize(image, (64, 64))  # Resize về (64, 64)
    image = image / 255.0  # Chuẩn hóa pixel về [0, 1]
    image = np.expand_dims(image, axis=-1)  # Thêm kênh màu (shape: 64, 64, 1)
    return image


# Lấy 1.000 ảnh từ mỗi thư mục con
def load_sampled_dataset(dataset_dir, sample_size=1000):
    X = []  # Danh sách ảnh đã tiền xử lý
    y = []  # Danh sách nhãn (0: Drowsy, 1: Non_Drowsy)
    
    for label, class_name in enumerate(['Drowsy', 'Non_Drowsy']):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Thư mục {class_name} không tồn tại.")
            continue
        
        # Lấy danh sách ảnh và chọn ngẫu nhiên sample_size ảnh
        image_files = os.listdir(class_dir)
        sampled_files = random.sample(image_files, min(len(image_files), sample_size))
        
        for image_name in sampled_files:
            image_path = os.path.join(class_dir, image_name)
            try:
                image = preprocess_image(image_path)
                X.append(image)
                y.append(label)
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh: {image_path}. Lỗi: {e}")
    
    X = np.array(X)  # Chuyển danh sách ảnh thành numpy array
    y = np.array(y)  # Chuyển danh sách nhãn thành numpy array
    return X, y

# Tải dữ liệu mẫu
X_sample, y_true_sample = load_sampled_dataset(dataset_dir, sample_size=1000)

# Thêm batch dimension cho X_sample (vì mô hình yêu cầu batch input)
X_sample = np.expand_dims(X_sample, axis=-1)  # Thêm kênh màu (gray -> (64, 64, 1))
X_sample = np.array(X_sample)
# Dự đoán với mô hình
y_pred_probs = model.predict(X_sample)  # X_sample shape: (batch_size, 64, 64, 1)
y_pred_sample = (y_pred_probs > 0.5).astype("int32").flatten()  # Chuyển xác suất thành nhãn (0 hoặc 1)

# Đánh giá kết quả
accuracy = accuracy_score(y_true_sample, y_pred_sample)
precision = precision_score(y_true_sample, y_pred_sample)
recall = recall_score(y_true_sample, y_pred_sample)
f1 = f1_score(y_true_sample, y_pred_sample)
conf_matrix = confusion_matrix(y_true_sample, y_pred_sample)
class_report = classification_report(y_true_sample, y_pred_sample, target_names=['Drowsy', 'Non_Drowsy'])


# In kết quả
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
# In ra terminal
print("=== Detailed Classification Report ===")
print(class_report)

# Trực quan hóa kết quả
def plot_metrics():
    # Vẽ các chỉ số đánh giá
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)  # Giới hạn giá trị trục y từ 0 đến 1
    plt.title("Evaluation Metrics", fontsize=16)
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Metrics", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12, color='black')
    plt.show()

def plot_confusion_matrix():
    # Vẽ confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=['Non_Drowsy', 'Drowsy'], yticklabels=['Non_Drowsy', 'Drowsy'])
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

# Hiển thị biểu đồ
plot_metrics()
plot_confusion_matrix()
