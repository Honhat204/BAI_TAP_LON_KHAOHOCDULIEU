import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Bỏ qua các cảnh báo để giao diện console gọn gàng hơn
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid")

# =====================================================================
# 0. CHUẨN BỊ VÀ LÀM SẠCH DỮ LIỆU
# =====================================================================
print("--- ĐANG TẢI VÀ CHUẨN BỊ DỮ LIỆU ---")

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "winequality-red.csv")

if os.path.exists(csv_path):
    print(f"-> Đang đọc dữ liệu Vang Đỏ cục bộ từ: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    # Tự động phát hiện dấu phân cách (dấu phẩy hoặc chấm phẩy)
    separator = ';' if ';' in first_line else ','
    df_red = pd.read_csv(csv_path, sep=separator)
    
    # Xử lý lỗi gộp cột (khi toàn bộ dữ liệu bị dồn vào 1 cột duy nhất)
    if df_red.shape[1] == 1:
        raw_col = df_red.columns[0]
        actual_headers = raw_col.split(',') if ',' in raw_col else raw_col.split(';')
        df_fixed = df_red[raw_col].str.split(',' if ',' in raw_col else ';', expand=True)
        df_fixed.columns = actual_headers
        df_red = df_fixed.apply(pd.to_numeric, errors='coerce')
else:
    # Tự động tải từ nguồn UCI nếu chưa có file trên máy
    print("-> Không tìm thấy file. Đang tải dữ liệu Vang Đỏ chuẩn từ kho UCI...")
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df_red = pd.read_csv(url_red, sep=";")
    df_red.to_csv(csv_path, index=False)

# Xóa dòng trùng lặp và các giá trị khuyết thiếu (NaN)
df_red = df_red.drop_duplicates().dropna()
df_red["wine_type"] = "Red Wine"

try:
    # Tải thêm dữ liệu vang trắng để so sánh
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df_white = pd.read_csv(url_white, sep=";")
    df_white = df_white.drop_duplicates().dropna()
    df_white["wine_type"] = "White Wine"
    
    # Nối 2 bảng dữ liệu đỏ và trắng theo chiều dọc
    df_all = pd.concat([df_red, df_white], ignore_index=True)
    has_white_wine = True
except Exception:
    has_white_wine = False
    print("-> Lỗi kết nối mạng: Không thể tải dữ liệu Vang Trắng.")

print("\n" + "="*70)
print("PHẦN 1: 5 CÂU HỎI PHÂN TÍCH DỮ LIỆU")
print("="*70)

# --- Câu 1 ---
# Tính giá trị trung bình của cột 'quality'
mean_quality = df_red["quality"].mean()
print(f"\n[Câu 1] Chất lượng trung bình của rượu vang Đỏ là: {mean_quality:.2f}/10")

# --- Câu 2 & Câu 5 ---
# Tạo ma trận tương quan Pearson giữa các biến dạng số
numeric_cols = df_red.select_dtypes(include=[np.number]).columns
corr_matrix = df_red[numeric_cols].corr()

# Rút trích hệ số tương quan của các biến đối với cột 'quality'
quality_corr = corr_matrix["quality"].drop("quality")
strongest_pos = quality_corr.idxmax() # Tương quan thuận mạnh nhất
strongest_neg = quality_corr.idxmin() # Tương quan nghịch mạnh nhất

print(f"\n[Câu 2 & 5] Hệ số tương quan của các thuộc tính với Chất lượng (Quality):")
print(quality_corr.sort_values(ascending=False).to_string())
print(f"  => Ảnh hưởng TÍCH CỰC mạnh nhất (Tương quan thuận): {strongest_pos} ({quality_corr[strongest_pos]:.3f})")
print(f"  => Ảnh hưởng TIÊU CỰC mạnh nhất (Tương quan nghịch): {strongest_neg} ({quality_corr[strongest_neg]:.3f})")

# Vẽ Heatmap (biểu đồ nhiệt) cho ma trận tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Ma trận tương quan giữa các chỉ số hóa học")
plt.tight_layout()
plt.savefig("b1_ma_tran_tuong_quan.png")
plt.close()

# --- Câu 3 ---
print(f"\n[Câu 3] So sánh Vang Đỏ và Vang Trắng:")
if has_white_wine:
    mean_white = df_white["quality"].mean()
    print(f"  - Điểm trung bình Vang Đỏ: {mean_quality:.2f}")
    print(f"  - Điểm trung bình Vang Trắng: {mean_white:.2f}")
    print(f"  => Vang Trắng được đánh giá cao hơn một chút (chênh lệch {abs(mean_white - mean_quality):.2f} điểm).")
    
    # Vẽ biểu đồ đếm tần suất chất lượng theo loại rượu
    plt.figure()
    sns.countplot(x="quality", hue="wine_type", data=df_all, palette=["#C8102E", "#F4E8C1"])
    plt.title("Phân phối điểm chất lượng: Vang Đỏ vs Vang Trắng")
    plt.savefig("b2_so_sanh_do_trang.png")
    plt.close()
else:
    print("  => Không có dữ liệu Vang Trắng để đối chiếu.")

# --- Câu 4 ---
print(f"\n[Câu 4] Mối quan hệ giữa Cồn (Alcohol) và Chất lượng:")
print("  => Tương quan thuận rõ rệt: Nồng độ cồn càng cao, rượu thường được chấm điểm chất lượng càng cao.")

# Vẽ Boxplot để xem sự phân tán nồng độ cồn theo từng mức chất lượng
plt.figure()
sns.boxplot(x="quality", y="alcohol", data=df_red, palette="Reds")
plt.title("Sự biến thiên của Nồng độ cồn theo Điểm chất lượng")
plt.savefig("b3_alcohol_vs_quality.png")
plt.close()

print("\n" + "="*70)
print("PHẦN 2: 2 CÂU HỎI DỰ ĐOÁN VÀ PHÂN LỚP (MACHINE LEARNING)")
print("="*70)

# --- Câu 1 (Phân lớp): Rừng ngẫu nhiên (Random Forest) ---
print("\n[ML Câu 1] Mô hình Phân Lớp (Classification) - Random Forest")

# Hàm chuyển đổi điểm số thành các nhãn phân lớp phân loại rõ ràng
def categorize_quality(q):
    if q <= 4: return "Thấp"
    elif q <= 6: return "Trung bình"
    else: return "Cao"

df_red["quality_label"] = df_red["quality"].apply(categorize_quality)

# Tách biến độc lập (X) và biến phụ thuộc/mục tiêu (y)
X_class = df_red.drop(["quality", "quality_label", "wine_type"], axis=1)
y_class = df_red["quality_label"]

# Chia dữ liệu thành tập Huấn luyện (80%) và tập Kiểm tra (20%)
# stratify=y_class giúp giữ nguyên tỷ lệ các nhãn giữa 2 tập
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)

# Khởi tạo và huấn luyện mô hình Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Đưa ra dự đoán trên tập kiểm tra
y_pred = rf.predict(X_test)

# In báo cáo độ chính xác (Precision, Recall, F1-Score)
print("Báo cáo độ chính xác của mô hình:")
print(classification_report(y_test, y_pred))

# Vẽ Ma trận nhầm lẫn (Confusion Matrix)
labels = ["Thấp", "Trung bình", "Cao"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.title("Ma trận nhầm lẫn - Phân lớp Chất lượng")
plt.ylabel("Thực tế")
plt.xlabel("Dự đoán")
plt.savefig("b4_matrix_phan_lop.png")
plt.close()

# --- Câu 2 (Phân cụm): Thuật toán K-Means ---
print("\n[ML Câu 2] Mô hình Phân Cụm (Clustering) - K-Means")

# Chuẩn hóa dữ liệu về cùng thang đo (Rất quan trọng cho K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_class)

# Khởi tạo mô hình K-Means để gom dữ liệu thành 3 cụm
kmeans = KMeans(n_clusters=3, random_state=42)
df_red["cluster"] = kmeans.fit_predict(X_scaled) # Phân cụm và lưu kết quả vào cột 'cluster'

print("Đặc điểm trung bình của 3 cụm (nhóm) rượu mới được phát hiện:")
cluster_stats = df_red.groupby("cluster")[["alcohol", "volatile acidity", "pH", "quality"]].mean()
print(cluster_stats.to_string())

# Trực quan hóa kết quả phân cụm qua biểu đồ Scatter (điểm phân tán)
plt.figure()
sns.scatterplot(x="alcohol", y="volatile acidity", hue="cluster", palette="Set1", data=df_red, s=100)
plt.title("Phân cụm Rượu Vang theo Cồn và Axit dễ bay hơi")
plt.savefig("b5_phan_cum_kmeans.png")
plt.close()

print("\n" + "*"*70)
print("HOÀN TẤT! Đã tự động xuất 5 biểu đồ (.png) vào thư mục của bạn.")
print("*"*70)