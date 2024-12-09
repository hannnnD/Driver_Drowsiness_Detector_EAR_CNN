Tổng quan các file trong dự án



File chính chạy chương trình

**_ear_model.py_**
Chức năng: Xây dựng và huấn luyện mô hình EAR (Eye Aspect Ratio) để phát hiện trạng thái buồn ngủ của người lái xe.
Chi tiết: Kết hợp dữ liệu EAR với các tham số đầu vào để xác định tình trạng buồn ngủ.

**drowsiness_detect_cnn_model.py**
Chức năng: Triển khai mô hình CNN (Convolutional Neural Network) để phát hiện buồn ngủ dựa trên ảnh mắt.
Chi tiết: Chạy mô hình được huấn luyện để phát hiện trạng thái buồn ngủ qua phân tích ảnh.

**drowsy_detected_combine_ear_cnn.py**
Chức năng: Kết hợp EAR và mô hình CNN để cải thiện độ chính xác trong phát hiện trạng thái buồn ngủ.
Chi tiết: Tổng hợp thông tin từ cả hai mô hình để đưa ra kết quả cuối cùng.


File phụ để đo đạc và hiệu chỉnh tham số

**ear_find_best_threshold.py**
Chức năng: Tìm giá trị ngưỡng tối ưu cho EAR để phát hiện trạng thái buồn ngủ.

**ear_evaluate.py**
Chức năng: Đánh giá hiệu suất của mô hình EAR trên dữ liệu thử nghiệm.

**cnn_model_evaluate.py**
Chức năng: Đánh giá hiệu suất của mô hình CNN trên tập dữ liệu.

**ear_rate.py**
Chức năng: Tính toán tỉ lệ EAR trong thời gian thực dựa trên dữ liệu ảnh.

**check_cnn_model_struc.py**
Chức năng: Kiểm tra cấu trúc mô hình CNN để đảm bảo đúng định dạng trước khi huấn luyện.


File mô hình đã huấn luyện

**drowsiness_detection_cnn_model_v1.h5**
Phiên bản 1 của mô hình CNN phát hiện buồn ngủ. Với Epoch = 10, Batch = 32, Tiền xử lý ảnh RGB để train

**drowsiness_detection_cnn_model_v2.h5**
Phiên bản 2 của mô hình CNN. Với Epoch = 20, Batch = 32, Tiền xử lý ảnh RGB để train

**drowsiness_detection_cnn_model_v3.h5**
Phiên bản 3 của mô hình CNN. Với Epoch = 20, Batch = 64, Tiền xử lý ảnh RGB để train

**drowsiness_detection_cnn_model_v35.h5**
Phiên bản mới nhất với tối ưu hóa kiến trúc và dữ liệu huấn luyện. Với Epoch = 20, Batch = 32, Tiền xử lý ảnh RGB thành ảnh xám để train


Dữ liệu và tệp hỗ trợ

**ear_data_images.csv**
Dữ liệu EAR được tổng hợp từ ảnh.

**ear_data_images_with_blur.csv**
Dữ liệu EAR đã được làm mờ để thử nghiệm khả năng phát hiện trong điều kiện khó khăn.

**shape_predictor_68_face_landmarks.dat**
Tệp hỗ trợ để xác định các điểm đặc trưng trên khuôn mặt (dùng trong phát hiện EAR).
