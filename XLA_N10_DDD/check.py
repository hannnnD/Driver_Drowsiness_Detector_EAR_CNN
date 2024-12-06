import os

# Đường dẫn chính xác đến thư mục dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'Dataset_DriverDrowsines(DDD)')


# Kiểm tra sự tồn tại của đường dẫn
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
else:
    print(f"Dataset path exists: {dataset_path}")
