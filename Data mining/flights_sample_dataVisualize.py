import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Đọc file csv
df = pd.read_csv("flights_sample_cleaned.csv")


# Phân tích số lượng chuyến bay theo từng năm
# Đếm số chuyến bay theo năm
flights_by_year = df['Year'].value_counts().sort_index()

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(flights_by_year.index, flights_by_year.values, marker='o')
plt.title("Số lượng chuyến bay theo năm")
plt.xlabel("Năm")
plt.ylabel("Số chuyến bay")
plt.grid()
plt.xticks(ticks=flights_by_year.index, labels=flights_by_year.index.astype(int))
plt.show()



# Tỷ lệ chuyến bay bị hủy hoặc chuyển hướng
# Tính tỷ lệ chuyến bay
cancelled = df['CANCELLED'].value_counts()
diverted = df['DIVERTED'].value_counts()

# Vẽ biểu đồ tròn
labels = ['Không hủy', 'Hủy', 'Không chuyển hướng', 'Chuyển hướng']
sizes = [len(df[df['CANCELLED'] == 0]), len(df[df['CANCELLED'] == 1]),
         len(df[df['DIVERTED'] == 0]), len(df[df['DIVERTED'] == 1])]
colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99']

plt.figure(figsize=(10, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title("Tỷ lệ chuyến bay bị hủy hoặc chuyển hướng")
plt.show()



# # Mối quan hệ giữa độ trễ khởi hành và độ trễ hạ cánh
# Vẽ biểu đồ phân tán
plt.figure(figsize=(10, 6))
sns.scatterplot(x='DEP_DELAY', y='ARR_DELAY', data=df, alpha=0.5)
plt.title("Mối quan hệ giữa độ trễ khởi hành và hạ cánh")
plt.xlabel("Độ trễ khởi hành (phút)")
plt.ylabel("Độ trễ hạ cánh (phút)")
plt.grid()
plt.show()