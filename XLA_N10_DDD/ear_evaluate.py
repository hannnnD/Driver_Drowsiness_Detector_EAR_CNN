import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Đường dẫn đến file CSV
csv_path = r"d:\Download\Driver-Drowsiness-Detector-master\XLA_N10_DDD\ear_data_images_with_blur.csv"

# Đọc dữ liệu từ CSV
data = pd.read_csv(csv_path)

# Chia dữ liệu thành EAR (feature) và Label (target)
X = data["EAR"].values.reshape(-1, 1)  # EAR là đặc trưng
y = data["Label"].values  # Label là nhãn (0: Non Drowsy, 1: Drowsy)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình đơn giản (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán nhãn cho tập test
y_pred = model.predict(X_test)

# Đánh giá hiệu suất
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")
recall = recall_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

# Hiển thị kết quả
print("Model Performance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
