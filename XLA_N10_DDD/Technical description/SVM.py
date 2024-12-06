import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Đọc ảnh từ file
image_path = os.path.join(os.path.dirname(__file__), 'test.jpeg')
image = cv2.imread(image_path)

# Chuyển ảnh sang thang độ xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng HOG để trích xuất đặc trưng
features, hog_image = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)

# Hiển thị ảnh gốc và ảnh HOG
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
ax1.imshow(gray_image, cmap=plt.cm.gray)
ax1.set_title('Ảnh gốc')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Ảnh HOG')
plt.show()

# -----------------------------
# Sử dụng SVM để phân loại
# -----------------------------

# Giả định: Bạn đã có tập dữ liệu huấn luyện
# Tạo tập dữ liệu ví dụ (dữ liệu thực tế sẽ phức tạp hơn)
# Ở đây, giả định `features` là vector đặc trưng từ HOG, ta cần một tập mẫu nhiều ảnh
X = []
y = []  # nhãn: 0 hoặc 1 (giả sử đây là bài toán nhị phân)

# Đọc nhiều ảnh làm tập dữ liệu mẫu
# Bạn cần thay thế phần này bằng việc load các ảnh từ tập dữ liệu thực tế
for i in range(10):  # giả sử có 10 ảnh mẫu
    label = i % 2  # gán nhãn luân phiên 0 và 1
    img_path = os.path.join(os.path.dirname(__file__), 'test1.jpg')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    X.append(feature)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
clf = svm.SVC(kernel='linear')  # Sử dụng kernel tuyến tính
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình SVM: {accuracy * 100:.2f}%")

# -----------------------------
# Dự đoán trên ảnh đầu vào
# -----------------------------

# Sử dụng ảnh gốc `gray_image` làm input
input_feature, _ = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)
input_feature = input_feature.reshape(1, -1)  # Định dạng lại vector thành (1, số lượng đặc trưng)

# Dự đoán nhãn
prediction = clf.predict(input_feature)
print(f"Dự đoán nhãn của ảnh đầu vào: {prediction[0]}")
