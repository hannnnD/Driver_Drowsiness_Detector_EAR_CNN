import cv2
import os
import dlib
import numpy as np
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model

# Đường dẫn tệp tin
current_dir = os.path.dirname(__file__)
cnn_model_path = os.path.join(current_dir, 'drowsiness_detection_cnn_model_v35.h5')
shape_predictor_path = os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat')
input_image_path = os.path.join(current_dir, 'A0127.png')  # Đường dẫn ảnh đầu vào
output_image_path = os.path.join(current_dir, 'output.jpg')  # Đường dẫn ảnh đầu ra

# Load mô hình CNN và dlib
cnn_model = load_model(cnn_model_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Hàm tính EAR (Eye Aspect Ratio)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])  # Khoảng cách chiều dọc 1
    B = dist.euclidean(eye[2], eye[4])  # Khoảng cách chiều dọc 2
    C = dist.euclidean(eye[0], eye[3])  # Khoảng cách chiều ngang
    ear = (A + B) / (2.0 * C)
    return ear

# Hàm tiền xử lý ảnh để đưa vào CNN
def preprocess_image(face_roi):
    face_resized = cv2.resize(face_roi, (64, 64))
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=-1)  # Thêm chiều kênh
    face_expanded = np.expand_dims(face_expanded, axis=0)    # Thêm chiều batch
    return face_expanded

# Chỉ số của mắt trái và mắt phải
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Đọc ảnh đầu vào
image = cv2.imread(input_image_path)
if image is None:
    raise FileNotFoundError("Không tìm thấy ảnh đầu vào!")

# Tiền xử lý ảnh
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

# Nếu phát hiện khuôn mặt, xử lý từng khuôn mặt
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_roi = gray[y:y+h, x:x+w]

    # Tiền xử lý khuôn mặt cho CNN
    processed_face = preprocess_image(face_roi)
    cnn_prediction = cnn_model.predict(processed_face)[0][0]
    cnn_label = "Awake" if cnn_prediction < 0.5 else "Drowsy"

    # Phát hiện landmark khuôn mặt
    shape = predictor(gray, face)
    shape = np.array([[p.x, p.y] for p in shape.parts()])

    # Tính EAR cho mắt trái và mắt phải
    left_eye = shape[LEFT_EYE]
    right_eye = shape[RIGHT_EYE]
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0

    # Hiển thị EAR trên ảnh
    cv2.putText(image, f"EAR: {ear:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị dự đoán CNN trên ảnh
    cv2.putText(image, f"CNN: {cnn_label}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Vẽ khung bao quanh khuôn mặt
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Vẽ mắt và landmark trên ảnh
    for (ex, ey) in left_eye:
        cv2.circle(image, (ex, ey), 2, (0, 0, 255), -1)
    for (ex, ey) in right_eye:
        cv2.circle(image, (ex, ey), 2, (0, 0, 255), -1)

# Lưu và hiển thị ảnh đầu ra
cv2.imwrite(output_image_path, image)
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Ảnh đầu ra đã được lưu tại: {output_image_path}")
