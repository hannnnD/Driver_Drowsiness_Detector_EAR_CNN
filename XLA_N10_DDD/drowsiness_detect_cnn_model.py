import tensorflow as tf
import cv2
import numpy as np
import os
import pygame

# Đường dẫn đến mô hình đã huấn luyện
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'drowsiness_detection_cnn_model_v2.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Ngưỡng để phát cảnh báo
THRESHOLD = 0.5  # Ngưỡng xác suất từ 0.5 trở lên sẽ được coi là buồn ngủ

# Đường dẫn file âm thanh cảnh báo
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), 'audio', 'alert.wav')) # Đảm bảo có sẵn file mp3 hoặc wav tại đường dẫn này

# Hàm xử lý ảnh đầu vào
def preprocess_image(frame):
    IMG_HEIGHT, IMG_WIDTH = 64, 64  # Kích thước ảnh đầu vào của mô hình
    # Chuyển ảnh sang kích thước phù hợp và chuẩn hóa
    resized_frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    normalized_frame = resized_frame / 255.0  # Chuẩn hóa pixel [0, 255] -> [0, 1]
    return np.expand_dims(normalized_frame, axis=0)  # Thêm batch dimension

# Sử dụng Webcam
cap = cv2.VideoCapture(0)  # Dùng webcam (0 là thiết bị camera mặc định)
if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

print("Đang sử dụng mô hình để phát hiện trạng thái...")

try:
    while True:
        ret, frame = cap.read()  # Đọc khung hình từ webcam
        if not ret:
            print("Không thể đọc khung hình!")
            break

        # Hiển thị video
        cv2.imshow("Webcam - Drowsiness Detection", frame)

        # Tiền xử lý khung hình
        processed_frame = preprocess_image(frame)

        # Dự đoán trạng thái
        prediction = model.predict(processed_frame)[0][0]  # Xác suất buồn ngủ
        print(f"Điểm số buồn ngủ: {prediction:.2f}")

        # Nếu xác suất vượt ngưỡng, phát âm thanh cảnh báo
        if prediction >= THRESHOLD:
            print("Phát hiện buồn ngủ! Đang phát cảnh báo...")
            pygame.mixer.music.play()

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Dừng phát hiện...")

finally:
    cap.release()
    cv2.destroyAllWindows()
