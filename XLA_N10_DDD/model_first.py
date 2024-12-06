import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

# Đường dẫn tới bộ dữ liệu
current_dir = os.path.abspath(os.path.dirname(__file__))
train_dir = os.path.join(current_dir, 'Drowsy_dataset', 'train')
test_dir = os.path.join(current_dir, 'Drowsy_dataset', 'test')

# Tạo ImageDataGenerator để tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Tải dữ liệu từ thư mục
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Đầu ra nhị phân
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Huấn luyện mô hình
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# Lưu mô hình
model.save(os.path.join(current_dir, 'drowsy_detection_model.h5'))
print("Mô hình đã được lưu thành công!")

# Dự đoán trên dữ liệu kiểm tra
test_data.reset()
y_true = test_data.classes
y_pred = model.predict(test_data)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)

# Tính toán các chỉ số
print("Báo cáo phân loại:")
print(classification_report(y_true, y_pred, target_names=['NATURAL', 'DROWSY']))

# Đồ thị Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Đồ thị Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Tải lại mô hình để sử dụng
loaded_model = load_model(os.path.join(current_dir, 'ddm_v1.h5'))
print("Mô hình đã được tải lại thành công!")

# Kiểm tra dự đoán bằng mô hình đã tải
y_pred_loaded = loaded_model.predict(test_data)
y_pred_loaded = (y_pred_loaded > 0.5).astype(int).reshape(-1)

print("Báo cáo phân loại (mô hình đã tải):")
print(classification_report(y_true, y_pred_loaded, target_names=['NATURAL', 'DROWSY']))
