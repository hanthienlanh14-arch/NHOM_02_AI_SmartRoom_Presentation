"""
ĐỒ ÁN MÔN: TRÍ TUỆ NHÂN TẠO - BIỂU DIỄN VÀ GIẢI THUẬT
Đề tài: Xây dựng hệ thống nhận biết tự động trạng thái hiện diện của con người
trong phòng thông minh dựa trên biểu diễn tri thức cây quyết định và các giải thuật Học máy.

Hướng dẫn chạy:
1. Tải bộ dữ liệu 'Occupancy Detection Dataset' trên Kaggle:
   https://www.kaggle.com/datasets/saumitgp/occupancy-detection-dataset
2. Giải nén và đặt các file (datatraining.txt, datatest.txt) cùng thư mục với file code này.
3. Cài đặt thư viện (nếu chưa có):
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
4. Chạy script:
   python train_occupancy_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==============================================================================
# 1. CẤU HÌNH & NẠP DỮ LIỆU
# ==============================================================================
FEATURES_ALL = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
FEATURES_NO_LIGHT = ["Temperature", "Humidity", "CO2", "HumidityRatio"]
TARGET = "Occupancy"


def load_occupancy_data(train_path="datatraining.txt", test_path="datatest.txt"):
    """Đọc dữ liệu từ file .txt hoặc .csv chuẩn của UCI/Kaggle."""
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Không tìm thấy '{train_path}' hoặc '{test_path}'.\n"
            "Vui lòng tải Occupancy Detection Dataset từ Kaggle và đặt vào cùng thư mục."
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"[INFO] Số lượng mẫu huấn luyện (Train): {len(df_train)}")
    print(f"[INFO] Số lượng mẫu kiểm thử (Test):  {len(df_test)}")
    print("[INFO] Kiểm tra giá trị thiếu (Missing values):")
    print(df_train[FEATURES_ALL + [TARGET]].isnull().sum().to_dict())
    return df_train, df_test


# ==============================================================================
# 2. HÀM ĐÁNH GIÁ & SO SÁNH CÁC GIẢI THUẬT
# ==============================================================================
def evaluate_scenario(df_train, df_test, feature_cols, scenario_name="Kịch bản 1"):
    print("\n" + "=" * 70)
    print(f"ĐANG CHẠY: {scenario_name} | Đặc trưng: {feature_cols}")
    print("=" * 70)

    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET].values
    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET].values

    # Chuẩn hóa Z-Score cho k-NN (khoảng cách Euclid rất nhạy với thang đo)
    scaler_z = StandardScaler()
    X_train_scaled = scaler_z.fit_transform(X_train)
    X_test_scaled = scaler_z.transform(X_test)

    # Khởi tạo 4 giải thuật đối sánh
    models = {
        "Decision Tree (ID3/Entropy)": (
            DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42),
            False,  # Cây quyết định không bắt buộc chuẩn hóa, giữ giá trị gốc để rút luật dễ đọc
        ),
        "k-NN (k=5, Euclidean)": (
            KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
            True,   # Dùng dữ liệu đã chuẩn hóa Z-Score
        ),
        "Gaussian Naive Bayes": (
            GaussianNB(),
            True,
        ),
        "Random Forest (100 trees)": (
            RandomForestClassifier(n_estimators=100, random_state=42),
            False,
        ),
    }

    results = []
    trained_models = {}

    for name, (model, use_scaled) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append(
            {
                "Giải thuật": name,
                "Accuracy (%)": round(acc * 100, 2),
                "Precision (%)": round(prec * 100, 2),
                "Recall (%)": round(rec * 100, 2),
                "F1-Score (%)": round(f1 * 100, 2),
            }
        )
        trained_models[name] = model

    df_results = pd.DataFrame(results)
    print("\n--- BẢNG SO SÁNH HIỆU NĂNG ---")
    print(df_results.to_string(index=False))

    # Trích xuất tập luật IF-THEN từ Cây quyết định
    dt_model = trained_models["Decision Tree (ID3/Entropy)"]
    rules_text = export_text(dt_model, feature_names=feature_cols)
    print("\n--- TẬP LUẬT RÚT TRÍCH TỪ CÂY QUYẾT ĐỊNH (max_depth=3) ---")
    print(rules_text)

    return df_results, trained_models, scaler_z


# ==============================================================================
# 3. VẼ BIỂU ĐỒ CHO BÁO CÁO (CÂY QUYẾT ĐỊNH, SO SÁNH, MA TRẬN NHẦM LẪN)
# ==============================================================================
def save_report_charts(df_train, df_test, dt_model_s1, res_s1, res_s2):
    os.makedirs("charts", exist_ok=True)

    # Biểu đồ 1: Sơ đồ Cây quyết định (Kịch bản 1)
    plt.figure(figsize=(16, 9))
    plot_tree(
        dt_model_s1,
        feature_names=FEATURES_ALL,
        class_names=["Phòng trống (0)", "Có người (1)"],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Biểu diễn tri thức bằng Cây quyết định (Kịch bản đủ 5 cảm biến)", fontsize=14)
    plt.tight_layout()
    plt.savefig("charts/decision_tree_s1.png", dpi=200)
    plt.close()

    # Biểu đồ 2: So sánh Accuracy giữa Kịch bản 1 (5 cảm biến) và Kịch bản 2 (Bỏ Light)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(res_s1))
    width = 0.35

    plt.bar(x - width / 2, res_s1["Accuracy (%)"], width, label="Kịch bản 1 (Đủ 5 cảm biến)", color="#2563eb")
    plt.bar(x + width / 2, res_s2["Accuracy (%)"], width, label="Kịch bản 2 (Tắt đèn - Bỏ Light)", color="#f59e0b")

    plt.xticks(x, res_s1["Giải thuật"], rotation=15, ha="right")
    plt.ylabel("Độ chính xác - Accuracy (%)")
    plt.ylim(70, 102)
    plt.title("So sánh hiệu năng các giải thuật qua 2 kịch bản thực nghiệm")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("charts/model_comparison.png", dpi=200)
    plt.close()

    # Biểu đồ 3: Ma trận nhầm lẫn (Confusion Matrix) cho Decision Tree ở Kịch bản 1
    y_pred_dt = dt_model_s1.predict(df_test[FEATURES_ALL].values)
    cm = confusion_matrix(df_test[TARGET].values, y_pred_dt)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Phòng trống (0)", "Có người (1)"],
        yticklabels=["Phòng trống (0)", "Có người (1)"],
    )
    plt.xlabel("Dự đoán (Predicted)")
    plt.ylabel("Thực tế (Actual)")
    plt.title("Confusion Matrix - Decision Tree (Kịch bản 1)")
    plt.tight_layout()
    plt.savefig("charts/confusion_matrix_dt.png", dpi=200)
    plt.close()

    print("\n[INFO] Đã lưu toàn bộ biểu đồ vào thư mục 'charts/' để chèn vào báo cáo.")


# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
if __name__ == "__main__":
    df_train, df_test = load_occupancy_data("datatraining.txt", "datatest.txt")

    # Kịch bản 1: Đủ 5 cảm biến
    res_s1, models_s1, scaler_s1 = evaluate_scenario(
        df_train, df_test, FEATURES_ALL, "KỊCH BẢN 1 (ĐỦ 5 CẢM BIẾN)"
    )

    # Kịch bản 2: Bỏ cảm biến Ánh sáng (mô phỏng ban đêm / tắt đèn)
    res_s2, models_s2, scaler_s2 = evaluate_scenario(
        df_train, df_test, FEATURES_NO_LIGHT, "KỊCH BẢN 2 (TẮT ĐÈN - BỎ LIGHT)"
    )

    # Lưu biểu đồ báo cáo
    save_report_charts(
        df_train, df_test, models_s1["Decision Tree (ID3/Entropy)"], res_s1, res_s2
    )

    # Lưu gói mô hình để chạy giao diện Demo Streamlit
    artifact = {
        "models_s1": models_s1,
        "scaler_s1": scaler_s1,
        "models_s2": models_s2,
        "scaler_s2": scaler_s2,
        "features_all": FEATURES_ALL,
        "features_no_light": FEATURES_NO_LIGHT,
    }
    joblib.dump(artifact, "occupancy_model.pkl")
    print("[INFO] Đã lưu mô hình vào 'occupancy_model.pkl'. Sẵn sàng chạy app_occupancy_demo.py!")