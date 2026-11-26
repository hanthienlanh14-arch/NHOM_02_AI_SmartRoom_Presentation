"""
GIAO DIỆN DEMO HỆ THỐNG NHẬN BIẾT NGƯỜI TRONG PHÒNG THÔNG MINH (STREAMLIT)

Hướng dẫn chạy:
1. Chạy file 'train_occupancy_model.py' trước để tạo file 'occupancy_model.pkl'.
2. Cài đặt Streamlit (nếu chưa có):
   pip install streamlit joblib numpy scikit-learn
3. Khởi chạy giao diện web:
   python -m streamlit run app_occupancy_demo.py
"""

import os
import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Nhận biết người trong phòng thông minh",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Hệ thống Nhận biết Hiện diện Phòng Thông minh")
st.caption("Đồ án Trí tuệ nhân tạo: Biểu diễn tri thức Cây quyết định & Giải thuật Học máy")

MODEL_PATH = "occupancy_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Chưa tìm thấy tệp mô hình `{MODEL_PATH}`. "
        "Vui lòng chạy lệnh `python train_occupancy_model.py` trước để huấn luyện mô hình!"
    )
    st.stop()


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


artifact = load_artifact()

# ==============================================================================
# HÀM GIẢI THÍCH ĐƯỜNG ĐI SUY DIỄN TRÊN CÂY QUYẾT ĐỊNH
# ==============================================================================
def explain_decision_path(dt_model, feature_names, sample_array):
    """Truy vết các nút nhánh mà mẫu Input đã đi qua để tạo giải thích IF-THEN."""
    tree = dt_model.tree_
    node_indicator = dt_model.decision_path(sample_array)
    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    conditions = []
    for node_id in node_index:
        # Bỏ qua nút lá
        if tree.children_left[node_id] == tree.children_right[node_id]:
            continue
        feat_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        val = sample_array[0, feat_idx]
        if val <= threshold:
            op = "<="
        else:
            op = ">"
        conditions.append(
            f"`{feature_names[feat_idx]}` ({val:.4g}) {op} {threshold:.2f}"
        )
    return " **VÀ** ".join(conditions)


# ==============================================================================
# GIAO DIỆN NHẬP LIỆU (INPUT)
# ==============================================================================
st.sidebar.header("⚙️ Cấu hình Kịch bản & Giải thuật")
night_mode = st.sidebar.checkbox(
    "Chế độ ban đêm / Tắt đèn (Bỏ cảm biến Light)", value=False
)

if night_mode:
    models_dict = artifact["models_s2"]
    scaler = artifact["scaler_s2"]
    feature_names = artifact["features_no_light"]
else:
    models_dict = artifact["models_s1"]
    scaler = artifact["scaler_s1"]
    feature_names = artifact["features_all"]

selected_model_name = st.sidebar.selectbox(
    "Chọn giải thuật phân lớp:",
    list(models_dict.keys()),
    index=0,
)

st.subheader("1. Nhập thông số từ 5 cảm biến môi trường")

col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("🌡️ Nhiệt độ phòng (°C)", min_value=10.0, max_value=45.0, value=21.5, step=0.1)
    humidity = st.number_input("💧 Độ ẩm tương đối (%)", min_value=10.0, max_value=95.0, value=27.2, step=0.5)
    humidity_ratio = st.number_input(
        "🌫️ Tỷ số độ ẩm (HumidityRatio)", min_value=0.001, max_value=0.020, value=0.0047, step=0.0001, format="%.4f"
    )

with col2:
    light = st.number_input(
        "💡 Cường độ ánh sáng (Lux)",
        min_value=0.0,
        max_value=2000.0,
        value=426.0,
        step=10.0,
        disabled=night_mode,
        help="Bị vô hiệu hóa khi bật chế độ ban đêm / tắt đèn",
    )
    co2 = st.number_input("💨 Nồng độ CO2 (ppm)", min_value=300.0, max_value=3000.0, value=721.0, step=10.0)

# ==============================================================================
# DỰ ĐOÁN & HIỂN THỊ KẾT QUẢ (OUTPUT)
# ==============================================================================
if st.button("🔍 Kiểm tra trạng thái phòng", type="primary", use_container_width=True):
    if night_mode:
        X_input = np.array([[temperature, humidity, co2, humidity_ratio]])
    else:
        X_input = np.array([[temperature, humidity, light, co2, humidity_ratio]])

    model = models_dict[selected_model_name]

    # Áp dụng chuẩn hóa Z-Score nếu giải thuật là k-NN hoặc Naive Bayes
    if "k-NN" in selected_model_name or "Naive Bayes" in selected_model_name:
        X_eval = scaler.transform(X_input)
    else:
        X_eval = X_input

    pred = int(model.predict(X_eval)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_eval)[0][pred] * 100
    else:
        proba = 100.0

    st.divider()
    st.subheader("2. Kết quả nhận diện & Đề xuất điều khiển")

    if pred == 1:
        st.success(f"🟢 **PHÁT HIỆN CÓ NGƯỜI TRONG PHÒNG** (Độ tin cậy: **{proba:.1f}%**)")
        st.info("⚡ **Lệnh Smart Home đề xuất:** Duy trì hệ thống chiếu sáng, thông gió và điều hòa ở mức thoải mái.")
    else:
        st.warning(f"⚪ **PHÒNG ĐANG TRỐNG** (Độ tin cậy: **{proba:.1f}%**)")
        st.info("🌱 **Lệnh Smart Home đề xuất:** Tự động tắt đèn và chuyển điều hòa sang chế độ tiết kiệm điện (Eco).")

    # Luôn hiển thị giải thích từ Cây quyết định để làm nổi bật tính minh bạch tri thức
    dt_model = models_dict["Decision Tree (ID3/Entropy)"]
    rule_path = explain_decision_path(dt_model, feature_names, X_input)
    st.markdown("**🧠 Giải thích suy diễn theo luật Cây quyết định:**")
    st.markdown(f"> **NẾU** {rule_path} **THÌ** Kết luận: `{'Có người (1)' if int(dt_model.predict(X_input)[0]) == 1 else 'Phòng trống (0)'}`")