import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Đường dẫn tới thư mục dữ liệu
train_dir = os.path.join(os.path.dirname(__file__), 'Dataset_DriverDrowsines(DDD)')

# Thông tin về ảnh
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Kích thước ảnh đầu vào
BATCH_SIZE = 64

# 1. Tạo ImageDataGenerator để đọc và chuẩn hóa ảnh
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Chuẩn hóa pixel từ [0-255] -> [0-1]
    validation_split=0.2  # Tách 20% dữ liệu làm tập validation
)

# Tạo generator cho tập huấn luyện
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',  # Chuyển ảnh sang 1 kênh (grayscale)
    class_mode='binary',  # Vì bài toán là nhị phân (2 lớp)
    subset='training'
)

# Tạo generator cho tập validation
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',  # Chuyển ảnh sang 1 kênh (grayscale)
    class_mode='binary',
    subset='validation'
)

# 2. Xây dựng mô hình CNN
model = models.Sequential([
    # Layer Convolutional 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),  # Input shape 1 kênh
    layers.MaxPooling2D((2, 2)),

    # Layer Convolutional 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Layer Convolutional 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten và Fully Connected
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout để tránh overfitting
    layers.Dense(1, activation='sigmoid')  # 1 lớp đầu ra với sigmoid cho bài toán nhị phân
])

# 3. Compile mô hình
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Sử dụng binary_crossentropy cho bài toán nhị phân
    metrics=['accuracy']
)

# 4. Huấn luyện mô hình
EPOCHS = 20
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# 5. Lưu mô hình
model.save("drowsiness_detection_cnn_model_v3.h5")

print("Huấn luyện hoàn tất và mô hình đã được lưu!")
