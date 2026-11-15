# -*- coding: utf-8 -*-
"""Local web GUI for the Oxford Flowers 102 classifier."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODEL_PATH = OUTPUTS / "flower_classifier.joblib"

st.set_page_config(
    page_title="Flower Lens | Oxford 102",
    page_icon="🌼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #19352d; --muted: #718078; --leaf: #47775c; --cream: #f6f7f1; --line: #e5e9df; }
      .stApp, [data-testid="stAppViewContainer"] { background: #fbfcf8; color: var(--ink); }
      html, body, [class*="css"] { font-family: 'Aptos', 'Segoe UI', sans-serif; }
      [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
      [data-testid="stMarkdownContainer"] li, label, [data-testid="stWidgetLabel"] p,
      [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {
        color: #40594e !important;
      }
      .stDeployButton { display: none; }
      h1, h2, h3 { color: var(--ink); }
      .hero { padding: 2.1rem 2.4rem; border: 1px solid var(--line); border-radius: 24px;
        background: linear-gradient(120deg, #eef4e9 0%, #f8f5e9 58%, #fffaf0 100%); margin-bottom: 1.4rem; }
      .hero h1 { font-family: Georgia, serif; font-size: 2.65rem; margin: 0 0 .4rem; }
      .hero p { color: #5c7065; font-size: 1.04rem; margin: 0; }
      .eyebrow { color: var(--leaf); text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 700; }
      .result-card { background: white; border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.15rem; margin: .45rem 0; }
      .result-name { color: var(--ink); font-size: 1rem; font-weight: 700; }
      .result-rank { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; }
      [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: .9rem 1rem; }
      [data-testid="stSidebar"] { background: #f1f4ed; border-right: 1px solid #e3e9df; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #40594e !important;
      }
      [data-testid="stAlert"] { background: #edf4eb; border: 1px solid #dce9d8; }
      [data-testid="stAlert"] p { color: #294b39 !important; }
      [data-testid="stFileUploader"] section { background: #fff; border: 1px dashed #9db6a1; }
      [data-testid="stFileUploader"] section * { color: #40594e !important; }
      button[data-baseweb="tab"] { color: #586e62 !important; }
      button[data-baseweb="tab"][aria-selected="true"] { color: #2e6545 !important; }
      div.stButton > button[kind="primary"] { background: #47775c; border: 0; border-radius: 12px; font-weight: 700; }
      div.stButton > button[kind="primary"]:hover { background: #365f49; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang tải mô hình đã huấn luyện...")
def load_model_bundle(model_path: str):
    bundle = joblib.load(model_path)
    backbone_name = bundle.get("backbone_name", "resnet18")
    if backbone_name == "resnet50":
        backbone = models.resnet50(weights=None)
    else:
        backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    backbone.load_state_dict(bundle["backbone_state_dict"])
    backbone.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(bundle.get("img_size", 224)),
        transforms.ToTensor(),
        transforms.Normalize(bundle.get("mean", (0.485, 0.456, 0.406)),
                             bundle.get("std", (0.229, 0.224, 0.225))),
    ])
    return bundle, backbone, transform, device


def predict_image(image: Image.Image, bundle, backbone, transform, device):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = backbone(tensor).cpu().numpy()
    scaled = bundle["scaler"].transform(features)
    probabilities = bundle["model"].predict_proba(scaled)[0]
    top_indices = np.argsort(probabilities)[::-1][:5]
    return [(bundle["class_names"][int(i)], float(probabilities[i])) for i in top_indices]


def show_prediction(upload, bundle, backbone, transform, device):
    image = Image.open(upload).convert("RGB")
    predictions = predict_image(image, bundle, backbone, transform, device)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.image(image, caption=upload.name, use_container_width=True)
    with right:
        st.markdown('<div class="eyebrow">Kết quả nhận diện</div>', unsafe_allow_html=True)
        st.markdown(f"### {predictions[0][0].title()}")
        st.caption(f"Xác suất mô hình: {predictions[0][1]:.1%} · Top‑5 gợi ý")
        for rank, (name, probability) in enumerate(predictions, start=1):
            st.markdown(
                f'<div class="result-card"><span class="result-rank">#{rank}</span>'
                f'<span class="result-name"> &nbsp; {name.title()}</span></div>',
                unsafe_allow_html=True,
            )
            st.progress(probability, text=f"{probability:.1%}")


def render_results_dashboard():
    metrics_path = OUTPUTS / "ket_qua_test.csv"
    if not metrics_path.exists():
        st.info("Chưa có kết quả đánh giá. Hãy chạy `nhom19.py` trước để tạo mô hình và báo cáo.")
        return
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        st.info("Tệp kết quả đánh giá đang trống.")
        return

    best = metrics.sort_values("F1 (macro)", ascending=False).iloc[0]
    columns = st.columns(4)
    columns[0].metric("Mô hình tốt nhất", str(best["Mô hình"]))
    columns[1].metric("Accuracy", f"{best['Accuracy']:.1%}")
    columns[2].metric("Macro F1", f"{best['F1 (macro)']:.1%}")
    columns[3].metric("Recall macro", f"{best['Recall (macro)']:.1%}")

    st.markdown("#### So sánh các mô hình")
    display_metrics = metrics.set_index("Mô hình")[[
        "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"
    ]]
    st.dataframe(
        display_metrics.style.format("{:.1%}").highlight_max(axis=0, color="#e7f1e5"),
        use_container_width=True,
    )
    st.bar_chart(display_metrics, y=["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"])

    confusion_path = OUTPUTS / "confusion_matrix.png"
    distribution_path = OUTPUTS / "eda_phan_phoi_nhan.png"
    knn_cv_path = OUTPUTS / "knn_chon_k_cv.png"
    knn_cv_table_path = OUTPUTS / "knn_chon_k_cv.csv"
    report_path = OUTPUTS / "classification_report_best.csv"
    first, second = st.columns(2, gap="large")
    with first:
        if confusion_path.exists():
            st.markdown("#### Confusion matrix")
            st.image(str(confusion_path), use_container_width=True)
    with second:
        if distribution_path.exists():
            st.markdown("#### Phân bố ảnh theo lớp")
            st.image(str(distribution_path), use_container_width=True)

    if knn_cv_path.exists():
        st.markdown("#### Chọn K cho KNN bằng Stratified 5-fold CV")
        st.image(str(knn_cv_path), use_container_width=True)
    if knn_cv_table_path.exists():
        with st.expander("Xem điểm CV của từng K và khoảng cách"):
            cv_table = pd.read_csv(knn_cv_table_path)
            st.dataframe(
                cv_table.style.format({
                    "Accuracy CV trung bình": "{:.4f}",
                    "Độ lệch chuẩn CV": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    if report_path.exists():
        report = pd.read_csv(report_path, index_col=0)
        class_rows = report[~report.index.isin(["accuracy", "macro avg", "weighted avg"])]
        if "f1-score" in class_rows.columns:
            class_rows = class_rows.sort_values("f1-score").head(10)
            st.markdown("#### 10 lớp khó nhận dạng nhất")
            st.dataframe(class_rows[["precision", "recall", "f1-score", "support"]],
                         use_container_width=True)


if not MODEL_PATH.exists():
    st.error(f"Chưa tìm thấy model tại `{MODEL_PATH}`. Hãy chạy `nhom19.py` để huấn luyện và lưu model.")
    st.stop()

bundle, backbone, image_transform, device = load_model_bundle(str(MODEL_PATH))

with st.sidebar:
    st.markdown("## 🌿 Flower Lens")
    st.caption("Oxford Flowers 102 · Computer Vision")
    st.divider()
    st.markdown("**Bộ trích xuất đặc trưng**")
    st.write(bundle.get("backbone_name", "ResNet18").upper())
    st.markdown("**Bộ phân loại dùng khi dự đoán**")
    st.write("Logistic Regression")
    st.markdown("**Số loài hoa**")
    st.write(len(bundle["class_names"]))
    st.divider()
    st.caption("Mô hình nhận ảnh hoa và trả về 5 loài có xác suất cao nhất.")

st.markdown(
    '<div class="hero"><div class="eyebrow">Oxford Flowers 102</div>'
    '<h1>Nhận diện vẻ đẹp của hoa</h1>'
    '<p>Tải ảnh lên để khám phá loài hoa. Mô hình sẽ phân tích đặc trưng và đưa ra 5 dự đoán phù hợp nhất.</p></div>',
    unsafe_allow_html=True,
)

predict_tab, evaluation_tab = st.tabs(["🌸 Nhận diện hoa", "📊 Đánh giá mô hình"])

with predict_tab:
    st.markdown("### Thử với ảnh của bạn")
    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều ảnh", type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True, help="Ảnh được xử lý ngay trên máy local của bạn.",
    )
    if uploaded_files:
        for index, uploaded_file in enumerate(uploaded_files):
            if len(uploaded_files) > 1:
                st.markdown(f"#### Ảnh {index + 1}: {uploaded_file.name}")
            try:
                show_prediction(uploaded_file, bundle, backbone, image_transform, device)
            except Exception as error:
                st.error(f"Không thể đọc hoặc phân tích ảnh {uploaded_file.name}: {error}")
    else:
        st.info("Thêm ảnh ở trên để bắt đầu nhận diện.")

with evaluation_tab:
    st.markdown("### Kết quả thực nghiệm")
    st.caption("Các chỉ số bên dưới được tạo từ tập valid có nhãn trong quy trình huấn luyện.")
    render_results_dashboard()

st.divider()
st.caption("Đồ án Nhập môn Học máy · Đề tài 7 · Oxford Flowers 102")
