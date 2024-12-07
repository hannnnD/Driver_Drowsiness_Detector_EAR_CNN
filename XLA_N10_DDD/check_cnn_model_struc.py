import numpy as np
from tensorflow.keras.models import load_model
import os

model_path = os.path.join(os.path.dirname(__file__), 'drowsiness_detection_cnn_model_v2.h5')

# Tải mô hình
model = load_model(model_path)
print("Model loaded successfully!")

# Kiểm tra cấu trúc model
model.summary()

# Kiểm tra kích thước đầu vào của mô hình
input_shape = model.input_shape[1:]  # Bỏ kích thước batch
print("Input shape:", input_shape)

# Tạo một ảnh giả lập (batch size = 1, và pixel giá trị ngẫu nhiên từ 0-1)
dummy_image = np.random.rand(1, *input_shape).astype('float32')  # Normalized

# Dự đoán với ảnh giả lập
pred = model.predict(dummy_image)
print("Prediction:", pred)

# Gắn nhãn theo logic nhị phân hoặc đa lớp
if pred.shape[1] == 1:  # Nhị phân
    predicted_label = 'Drowsy' if pred[0][0] < 0.5 else 'Natural'
else:  # Đa lớp
    predicted_label_index = np.argmax(pred[0])
    predicted_label = f"Class {predicted_label_index}"

print("Predicted Label:", predicted_label)
