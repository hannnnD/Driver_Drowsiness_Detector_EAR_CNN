import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os
# Đọc ảnh từ file
image_path = os.path.join(os.path.dirname(__file__), 'test.jpeg')
image = cv2.imread(image_path)

# Chuyển ảnh sang thang độ xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Áp dụng HOG
features, hog_image = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)


# Chuẩn hóa ảnh HOG để hiển thị
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))


# Hiển thị ảnh gốc và ảnh HOG
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

ax1.imshow(gray_image, cmap=plt.cm.gray)
ax1.set_title('Ảnh gốc')

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Ảnh HOG')

plt.show()
