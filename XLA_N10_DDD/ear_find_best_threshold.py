import pandas as pd
import numpy as np

# Đường dẫn đến file CSV
file_path = r"d:\Download\Driver-Drowsiness-Detector-master\XLA_N10_DDD\ear_data_images_with_blur.csv"

# Đọc dữ liệu từ file CSV
data = pd.read_csv(file_path)

# Xác định khoảng giá trị EAR để thử nghiệm
ear_range = np.arange(0.15, 0.35, 0.01)

# Hàm tính độ chính xác
def calculate_accuracy(data, threshold):
    correct_predictions = 0
    for _, row in data.iterrows():
        ear = row["EAR"]
        label = row["Label"]
        # Dự đoán dựa trên ngưỡng EAR
        predicted_label = 1 if ear < threshold else 0
        if predicted_label == label:
            correct_predictions += 1
    accuracy = correct_predictions / len(data)
    return accuracy

# Tìm ngưỡng tốt nhất
best_threshold = None
best_accuracy = 0

print("Testing thresholds...")
for threshold in ear_range:
    accuracy = calculate_accuracy(data, threshold)
    print(f"Threshold: {threshold:.6f}, Accuracy: {accuracy:.6f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

# In ngưỡng tốt nhất
print(f"\nBest EAR threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.4f}")

# Lưu kết quả ra file
output_path = "best_threshold_results.csv"
with open(output_path, "w") as file:
    file.write(f"Best EAR Threshold,Accuracy\n{best_threshold:.2f},{best_accuracy:.4f}\n")

print(f"Results saved to {output_path}")
