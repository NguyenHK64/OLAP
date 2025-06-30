import pandas as pd
import matplotlib.pyplot as plt

# Đọc file csv
df = pd.read_csv("flights_sample_3m.csv")

# Kiểm tra số lượng dòng và cột
num_rows, num_columns = df.shape
print(f"Số lượng dòng: {num_rows}")
print(f"Số lượng cột: {num_columns}")

# Kiểm tra số lượng dòng bị trùng lặp
duplicate_rows = df.duplicated().sum()
print(f"Số lượng dòng bị trùng lặp: {duplicate_rows}")

# Kiểm tra chi tiết số lượng giá trị thiếu theo từng cột
missing_per_column = df.isnull().sum()
print("Số lượng giá trị thiếu theo từng cột:")
print(missing_per_column)

# Danh sách các cột cần xóa
columns_to_drop = [
    "DELAY_DUE_CARRIER",
    "DELAY_DUE_WEATHER",
    "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY",
    "DELAY_DUE_LATE_AIRCRAFT",
    "CANCELLATION_CODE"
]

# Xóa các cột được chỉ định
df.drop(columns=columns_to_drop, inplace=True)

# Xóa các cột có hơn 60% dữ liệu bị thiếu
threshold = 0.6  # Ngưỡng 60%
missing_percent = df.isnull().mean()  # Tỷ lệ thiếu dữ liệu mỗi cột
columns_to_remove = missing_percent[missing_percent > threshold].index  # Các cột cần xóa
df.drop(columns=columns_to_remove, inplace=True)

# Kiểm tra lại DataFrame sau khi xóa
print("Các cột còn lại:")
print(df.columns)

# Chuyển đổi cột FL_DATE sang kiểu datetime
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format='%Y-%m-%d')

# Tách FL_DATE thành Ngày, Tháng, Năm
df['Day'] = df['FL_DATE'].dt.day    # Ngày
df['Month'] = df['FL_DATE'].dt.month  # Tháng
df['Year'] = df['FL_DATE'].dt.year   # Năm

# Kiểm tra kết quả
print(df[['FL_DATE', 'Day', 'Month', 'Year']].head())

# Lưu dataset sau khi xử lí vào frame mới
df.to_csv("flights_sample_cleaned.csv", index=False)


