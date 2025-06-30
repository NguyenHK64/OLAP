import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import joblib

# Đọc dữ liệu
df = pd.read_csv("flights_sample_cleaned.csv")

# Chọn các cột cần thiết
columns_to_use = [
    'FL_DATE', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'ORIGIN', 'DEST', 
    'AIRLINE', 'DEP_DELAY', 'DISTANCE', 'TAXI_OUT', 'TAXI_IN', 'ARR_DELAY'
]
df = df[columns_to_use]

# Loại bỏ các hàng có giá trị thiếu
df.dropna(subset=['ARR_DELAY'], inplace=True)

# Lấy mẫu dữ liệu (100.000 dòng)
df_sample = df.sample(n=100000, random_state=42)  # Lấy 100.000 dòng dữ liệu ngẫu nhiên

# Chuyển đổi thời gian
df_sample['FL_DATE'] = pd.to_datetime(df_sample['FL_DATE'], format='%Y-%m-%d')
df_sample['Year'] = df_sample['FL_DATE'].dt.year
df_sample['Month'] = df_sample['FL_DATE'].dt.month
df_sample['Day'] = df_sample['FL_DATE'].dt.day
df_sample['CRS_DEP_MIN'] = (df_sample['CRS_DEP_TIME'] // 100) * 60 + (df_sample['CRS_DEP_TIME'] % 100)
df_sample['CRS_ARR_MIN'] = (df_sample['CRS_ARR_TIME'] // 100) * 60 + (df_sample['CRS_ARR_TIME'] % 100)
df_sample.drop(columns=['FL_DATE', 'CRS_DEP_TIME', 'CRS_ARR_TIME'], inplace=True)

# Tách đặc trưng và mục tiêu
categorical_features = ['AIRLINE', 'ORIGIN', 'DEST']
numerical_features = ['DEP_DELAY', 'DISTANCE', 'TAXI_OUT', 'TAXI_IN', 'CRS_DEP_MIN', 'CRS_ARR_MIN', 'Year', 'Month', 'Day']
X = df_sample.drop(columns=['ARR_DELAY'])
y = df_sample['ARR_DELAY']

# Xử lý dữ liệu
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_preprocessed = pipeline.fit_transform(X)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
print(f"Số lượng mẫu trong tập huấn luyện: {X_train.shape[0]}")
print(f"Số lượng mẫu trong tập kiểm tra: {X_test.shape[0]}")

# Huấn luyện mô hình XGBoost
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Kết quả dự đoán
results = pd.DataFrame({
    'Thực tế': y_test.values,
    'Dự đoán': y_pred
})

# Lưu kết quả vào file CSV
results.to_csv("predicted_results_xgboost_100000.csv", index=False)
print("Kết quả dự đoán đã được lưu vào file 'predicted_results_xgboost_100000.csv'")

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Tính RMSE bằng numpy.sqrt
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")



import matplotlib.pyplot as plt

# Biểu đồ So sánh Giá trị Thực tế và Dự đoán
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k', label="Dự đoán")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Đường y = x")
plt.title("So sánh Giá trị Thực tế và Dự đoán")
plt.xlabel("Giá trị Thực tế (phút)")
plt.ylabel("Giá trị Dự đoán (phút)")
plt.legend()
plt.grid(True)
plt.show()
