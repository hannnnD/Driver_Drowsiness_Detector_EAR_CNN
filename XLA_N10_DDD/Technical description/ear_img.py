import cv2
import os
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Hàm tính toán EAR
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Hàm lọc Gaussian
def apply_gaussian_blur(gray, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(gray, kernel_size, sigma)

# Các chỉ số cho mô hình
EAR_THRESHOLD = 0.3

# Đường dẫn đến file ảnh và mô hình
image_path = os.path.join(os.path.dirname(__file__), 'test.jpeg')
shape_predictor_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')

# Tải mô hình nhận diện khuôn mặt và landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Đọc ảnh đầu vào
frame = cv2.imread(image_path)
frame_original = frame.copy()  # Lưu bản gốc để xuất ảnh đầu vào
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 1. Tiền xử lý ảnh
gray_blurred = apply_gaussian_blur(gray, kernel_size=(5, 5), sigma=1)

# 2. Trích xuất đặc trưng
faces = detector(gray_blurred)
frame_features = frame.copy()  # Dùng để vẽ đặc trưng

for face in faces:
    # Lấy tọa độ khuôn mặt
    shape = predictor(gray_blurred, face)
    shape = np.array([[p.x, p.y] for p in shape.parts()])

    # Vùng mắt
    left_eye = shape[36:42]
    right_eye = shape[42:48]

    # Tính EAR cho từng mắt
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    ear = (left_ear + right_ear) / 2.0

    # Vẽ đặc trưng mắt lên ảnh
    left_eye_hull = cv2.convexHull(left_eye)
    right_eye_hull = cv2.convexHull(right_eye)
    cv2.drawContours(frame_features, [left_eye_hull], -1, (0, 255, 0), 2)  # Màu xanh lá
    cv2.drawContours(frame_features, [right_eye_hull], -1, (0, 255, 0), 2)

    # Vẽ tất cả các landmarks khuôn mặt
    for (x, y) in shape:
        cv2.circle(frame_features, (x, y), 2, (255, 0, 0), -1)  # Màu xanh dương

    # Vẽ vùng mặt
    cv2.rectangle(frame_features, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 255), 2)  # Màu vàng

    # 3. Phân loại trạng thái
    if ear < EAR_THRESHOLD:
        cv2.putText(frame, "Eyes Closed", (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Tô đỏ vùng mắt khi nhắm
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), -1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), -1)
    else:
        cv2.putText(frame, "Eyes Open", (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Tô xanh vùng mắt khi mở
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), -1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), -1)

# Xuất ảnh
cv2.imwrite("output_1_original.jpg", frame_original)  # Ảnh đầu vào
cv2.imwrite("output_2_preprocessed.jpg", gray_blurred)  # Ảnh tiền xử lý
cv2.imwrite("output_3_features.jpg", frame_features)  # Ảnh trích xuất đặc trưng
cv2.imwrite("output_4_classified.jpg", frame)  # Ảnh phân loại trạng thái

print("Các ảnh đã được lưu thành công:")
print("- output_1_original.jpg: Ảnh đầu vào")
print("- output_2_preprocessed.jpg: Ảnh tiền xử lý")
print("- output_3_features.jpg: Ảnh trích xuất đặc trưng")
print("- output_4_classified.jpg: Ảnh phân loại trạng thái")
