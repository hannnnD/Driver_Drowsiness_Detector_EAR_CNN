import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Đường dẫn tới mô hình đã huấn luyện và ảnh đầu vào
model_path = os.path.join(os.path.dirname(__file__), 'drowsiness_detection_cnn_model_v35.h5')
image_path = os.path.join(os.path.dirname(__file__), 'A0127.png')

# Kích thước ảnh đầu vào cho mô hình
IMG_HEIGHT, IMG_WIDTH = 64, 64

# Tải mô hình đã huấn luyện
model = load_model(model_path)

# Đọc ảnh đầu vào
image = cv2.imread(image_path)
image_original = image.copy()  # Lưu bản gốc để hiển thị cuối cùng

# Tiền xử lý ảnh: Chuyển sang grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tiền xử lý ảnh: GaussianBlur
image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 1)

# Resize ảnh về kích thước đầu vào của mô hình
image_resized = cv2.resize(image_blurred, (IMG_WIDTH, IMG_HEIGHT))
image_normalized = image_resized.astype("float32") / 255.0  # Chuẩn hóa pixel về [0, 1]
image_expanded = np.expand_dims(image_normalized, axis=(0, -1))  # Thêm batch và channel dimension

# Dự đoán nhãn bằng mô hình
prediction = model.predict(image_expanded)[0][0]

# Gắn nhãn dự đoán
label = "Non Drowsy" if prediction >= 0.5 else "Drowsy"
color = (0, 255, 0) if label == "Non Drowsy" else (0, 0, 255)

# Vẽ nhãn lên ảnh gốc
cv2.putText(image_original, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Lưu ảnh tiền xử lý và ảnh kết quả
cv2.imwrite("output_preprocessed.jpg", image_blurred)  # Ảnh đã tiền xử lý
cv2.imwrite("output_prediction.jpg", image_original)  # Ảnh có nhãn dự đoán

# Hiển thị ảnh đầu ra
cv2.imshow("Preprocessed Image", image_blurred)  # Ảnh tiền xử lý
cv2.imshow("Prediction", image_original)  # Ảnh kết quả
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Ảnh đã được lưu:")
print("- output_preprocessed.jpg: Ảnh tiền xử lý")
print("- output_prediction.jpg: Ảnh có nhãn dự đoán")
