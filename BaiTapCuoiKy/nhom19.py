# -*- coding: utf-8 -*-
"""Phân loại Oxford Flowers 102 bằng đặc trưng transfer learning và ML cổ điển.

Chạy local:  py -m pip install -r requirements.txt
             py nhom19.py

Lần chạy đầu cần tải trọng số ImageNet của backbone; các đặc trưng ảnh được
lưu trong outputs/cache để những lần chạy sau không phải trích xuất lại.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Chạy được cả trên máy không cài giao diện đồ họa.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset" / "flower_data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá phân loại hoa Oxford 102")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Thư mục chứa train/, valid/ và cat_to_name.json")
    parser.add_argument("--backbone", choices=("resnet18", "resnet50"), default="resnet18",
                        help="ResNet18 chạy nhanh hơn trên CPU; ResNet50 cho phép so sánh")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augment", type=int, default=0,
                        help="Số lượt ảnh augmentation bổ sung cho mỗi ảnh train (mặc định 0 để chạy nhanh)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Số tiến trình đọc ảnh; để 0 là lựa chọn ổn định trên Windows")
    parser.add_argument("--force-features", action="store_true",
                        help="Trích xuất lại đặc trưng, bỏ qua cache đã lưu")
    return parser.parse_args()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_paths(folder):
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def discover_dataset(data_dir):
    data_dir = data_dir.resolve()
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    if not train_dir.is_dir() or not valid_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy train/ và valid/ trong {data_dir}. "
            "Hãy truyền đúng thư mục bằng --data."
        )

    rows = []
    for split, split_dir in (("train", train_dir), ("valid", valid_dir)):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in image_paths(class_dir):
                rows.append({"path": str(path), "class_folder": class_dir.name, "split": split})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Không tìm thấy ảnh trong {data_dir}")

    classes = sorted(frame["class_folder"].unique(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    folder_to_label = {name: i for i, name in enumerate(classes)}
    frame["label"] = frame["class_folder"].map(folder_to_label).astype(int)

    names_path = data_dir / "cat_to_name.json"
    if names_path.exists():
        raw_names = json.loads(names_path.read_text(encoding="utf-8"))
        class_names = [raw_names.get(name, name) for name in classes]
    else:
        class_names = classes

    # Dùng Stratified 5-fold CV trên toàn bộ train để chọn K; giữ valid/ nguyên
    # vẹn làm tập đánh giá cuối cùng.
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    test = frame[frame["split"] == "valid"].reset_index(drop=True)

    unlabeled_test = image_paths(data_dir / "test") if (data_dir / "test").is_dir() else []
    print(f"Dataset: {data_dir}")
    print(f"Ảnh train: {len(train):,} | valid (đánh giá cuối): {len(test):,} | lớp: {len(classes)}")
    if unlabeled_test:
        print(f"Thư mục test không nhãn: {len(unlabeled_test):,} ảnh (chỉ dùng demo dự đoán).")
    return frame, train, test, unlabeled_test, classes, class_names


def make_eda(frame, train, class_names):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = frame[frame["split"] == "train"]["label"].value_counts().sort_index()
    plt.figure(figsize=(16, 5))
    plt.bar(np.arange(len(counts)), counts.values, color="tab:purple")
    plt.title("Số ảnh train theo lớp hoa")
    plt.xlabel("Mã lớp")
    plt.ylabel("Số ảnh")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_phan_phoi_nhan.png", dpi=130)
    plt.close()

    sample_idx = np.linspace(0, len(train) - 1, min(8, len(train)), dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, idx in zip(axes.ravel(), sample_idx):
        row = train.iloc[idx]
        with Image.open(row["path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(class_names[int(row["label"])], fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(sample_idx):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_anh_mau.png", dpi=130)
    plt.close(fig)

    sizes = []
    for path in tqdm(frame["path"], desc="Đọc kích thước ảnh"):
        with Image.open(path) as image:
            sizes.append((image.width, image.height, len(image.getbands())))
    size_frame = pd.DataFrame(sizes, columns=["width", "height", "channels"])
    print("Kích thước ảnh (width, height):")
    print(size_frame[["width", "height"]].describe().round(1))
    print("Số kênh:", size_frame["channels"].value_counts().to_dict())


def build_transforms():
    evaluate = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    augment = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return evaluate, augment


class FlowerDataset(Dataset):
    def __init__(self, frame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(self.labels[index])


def make_backbone(name, device):
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_features(frame, model, transform, device, args, description):
    loader = DataLoader(
        FlowerDataset(frame, transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    features, labels = [], []
    for images, batch_labels in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        features.append(model(images).cpu().numpy().astype(np.float32, copy=False))
        labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)


def load_or_extract(args, train, test, device):
    evaluate, augment = build_transforms()
    model = make_backbone(args.backbone, device)
    cache_dir = OUTPUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature_source = "\n".join(
        f"{split}:{row.path}:{row.label}"
        for split, part in (("train", train), ("test", test))
        for row in part.itertuples()
    )
    data_signature = hashlib.sha1(signature_source.encode("utf-8")).hexdigest()[:10]
    cache_path = cache_dir / (
        f"features_cv5_{args.backbone}_size{IMAGE_SIZE}_aug{args.augment}_{data_signature}.npz"
    )

    if cache_path.exists() and not args.force_features:
        cached = np.load(cache_path)
        print(f"Dùng cache đặc trưng: {cache_path}")
        keys = ("X_train", "y_train", "X_cv", "y_cv", "X_test", "y_test")
        return tuple(cached[k] for k in keys), model

    start = time.time()
    # Lưu riêng đặc trưng ảnh gốc để CV không bị rò rỉ giữa một ảnh và các
    # phiên bản augmentation của chính ảnh đó.
    X_cv, y_cv = extract_features(train, model, evaluate, device, args, "Trích xuất train")
    train_features, train_labels = [X_cv], [y_cv]
    for i in range(args.augment):
        x, y = extract_features(train, model, augment, device, args, f"Augmentation {i + 1}/{args.augment}")
        train_features.append(x)
        train_labels.append(y)

    x_test, y_test = extract_features(test, model, evaluate, device, args, "Đánh giá")
    arrays = (
        np.concatenate(train_features), np.concatenate(train_labels),
        X_cv, y_cv, x_test, y_test,
    )
    np.savez_compressed(cache_path, X_train=arrays[0], y_train=arrays[1],
                        X_cv=arrays[2], y_cv=arrays[3], X_test=arrays[4], y_test=arrays[5])
    print(f"Đã lưu cache: {cache_path} ({time.time() - start:.0f} giây)")
    return arrays, model


def fit_models(arrays):
    X_train, y_train, X_cv, y_cv, X_test, y_test = arrays

    # Stratified 5-fold CV trên ảnh train gốc. Scaler được fit riêng trong
    # từng fold để không đưa thống kê của fold validation vào fold train.
    k_values = np.arange(1, 30, 2)
    metrics = ("euclidean", "manhattan", "cosine")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = {
        metric: np.zeros((cv.get_n_splits(), len(k_values)), dtype=np.float32)
        for metric in metrics
    }
    n_classes = len(np.unique(y_cv))
    k_position = {int(k): i for i, k in enumerate(k_values)}

    print("\nChọn K bằng Stratified 5-fold CV (K lẻ từ 1 đến 29)...")
    for fold_index, (fit_indices, valid_indices) in enumerate(cv.split(X_cv, y_cv)):
        fold_scaler = StandardScaler()
        X_fold_train = fold_scaler.fit_transform(X_cv[fit_indices])
        X_fold_valid = fold_scaler.transform(X_cv[valid_indices])
        y_fold_train = y_cv[fit_indices]
        y_fold_valid = y_cv[valid_indices]
        row_indices = np.arange(len(valid_indices))

        for metric in metrics:
            neighbors = KNeighborsClassifier(
                n_neighbors=int(k_values[-1]), metric=metric, n_jobs=-1
            )
            neighbors.fit(X_fold_train, y_fold_train)
            nearest_indices = neighbors.kneighbors(X_fold_valid, return_distance=False)
            nearest_labels = y_fold_train[nearest_indices]
            votes = np.zeros((len(valid_indices), n_classes), dtype=np.uint8)

            for rank in range(int(k_values[-1])):
                votes[row_indices, nearest_labels[:, rank]] += 1
                k = rank + 1
                if k in k_position:
                    fold_scores[metric][fold_index, k_position[k]] = np.mean(
                        votes.argmax(axis=1) == y_fold_valid
                    )

    cv_means = {metric: scores.mean(axis=0) for metric, scores in fold_scores.items()}
    cv_stds = {metric: scores.std(axis=0, ddof=1) for metric, scores in fold_scores.items()}
    best_by_metric = {}
    for metric in metrics:
        best_index = int(np.argmax(cv_means[metric]))
        best_by_metric[metric] = (float(cv_means[metric][best_index]), int(k_values[best_index]))
        print(f"{metric:9s}: K tốt nhất = {k_values[best_index]}, "
              f"CV acc = {cv_means[metric][best_index]:.4f}")

    cv_table = pd.DataFrame([
        {
            "Khoảng cách": metric,
            "K": int(k),
            "Accuracy CV trung bình": float(cv_means[metric][index]),
            "Độ lệch chuẩn CV": float(cv_stds[metric][index]),
        }
        for metric in metrics
        for index, k in enumerate(k_values)
    ])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(OUTPUT_DIR / "knn_chon_k_cv.csv", index=False, encoding="utf-8-sig")

    colors = {"euclidean": "tab:blue", "manhattan": "tab:orange", "cosine": "tab:green"}
    plt.figure(figsize=(9, 5))
    for metric in metrics:
        mean = cv_means[metric]
        std = cv_stds[metric]
        plt.plot(k_values, mean, marker="o", linewidth=2, markersize=5,
                 color=colors[metric], label=metric)
        plt.fill_between(k_values, mean - std, mean + std, color=colors[metric], alpha=0.18)
    plt.title("Chọn K cho KNN bằng Stratified 5-fold CV")
    plt.xlabel("K")
    plt.ylabel("Accuracy (5-fold CV)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knn_chon_k_cv.png", dpi=150)
    plt.close()

    best_metric = max(metrics, key=lambda metric: best_by_metric[metric][0])
    best_cv_score, best_k = best_by_metric[best_metric]
    print(f"=> Chọn KNN: K = {best_k}, khoảng cách = {best_metric} "
          f"(CV acc = {best_cv_score:.4f})")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn_name = f"KNN (K={best_k}, {best_metric})"
    knn_model = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, n_jobs=-1)

    models_to_score = {
        knn_name: knn_model,
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
        # Linear SVM với hinge loss, huấn luyện bằng SGD nhanh hơn liblinear
        # đáng kể khi phải giải bài toán 102 lớp theo chiến lược one-vs-rest.
        "SVM tuyến tính (SGD)": SGDClassifier(
            loss="hinge", alpha=1e-4, max_iter=500, tol=1e-3,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=5, random_state=SEED, n_jobs=-1,
        ),
    }

    results, predictions = [], {}
    for name, model in models_to_score.items():
        print(f"Huấn luyện/đánh giá: {name}")
        model.fit(X_train, y_train)
        start = time.time()
        prediction = model.predict(X_test)
        elapsed = time.time() - start
        predictions[name] = prediction
        results.append({
            "Mô hình": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision (macro)": precision_score(y_test, prediction, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, prediction, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, prediction, average="macro", zero_division=0),
            "Thời gian dự đoán (s)": elapsed,
        })

    result_frame = pd.DataFrame(results).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    return scaler, models_to_score, result_frame, predictions, y_test, X_test


def save_results(class_names, models_to_score, scaler, results, predictions, y_test, model_backbone,
                 unlabeled_test, device, args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "ket_qua_test.csv", index=False, encoding="utf-8-sig")
    print("\nKết quả trên tập valid gốc:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    plot_data = results.melt(id_vars="Mô hình", value_vars=[
        "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"
    ], var_name="Chỉ số", value_name="Giá trị")
    plt.figure(figsize=(11, 5))
    sns.barplot(data=plot_data, x="Mô hình", y="Giá trị", hue="Chỉ số")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "so_sanh_mo_hinh.png", dpi=140)
    plt.close()

    best_name = results.iloc[0]["Mô hình"]
    best_model = models_to_score[best_name]
    y_best = predictions[best_name]
    report = pd.DataFrame(classification_report(
        y_test, y_best, labels=np.arange(len(class_names)), target_names=class_names,
        output_dict=True, zero_division=0,
    )).T
    report.to_csv(OUTPUT_DIR / "classification_report_best.csv", encoding="utf-8-sig")
    print(f"\nMô hình tốt nhất: {best_name}")
    print("10 lớp có F1 thấp nhất:")
    print(report.iloc[:len(class_names)].sort_values("f1-score").head(10)[["precision", "recall", "f1-score", "support"]])

    cm = confusion_matrix(y_test, y_best, labels=np.arange(len(class_names)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(13, 11))
    sns.heatmap(cm_norm, cmap="Blues", vmin=0, vmax=1, xticklabels=False, yticklabels=False)
    plt.title(f"Confusion matrix chuẩn hóa – {best_name}")
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thật")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=140)
    plt.close()

    bundle = {
        "scaler": scaler,
        "model": models_to_score["Logistic Regression"],
        "class_names": class_names,
        "img_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
        "backbone_state_dict": {key: value.cpu() for key, value in model_backbone.state_dict().items()},
        "backbone_name": args.backbone,
    }
    joblib.dump(bundle, OUTPUT_DIR / "flower_classifier.joblib", compress=3)

    if unlabeled_test:
        demo_frame = pd.DataFrame({"path": [str(p) for p in unlabeled_test], "label": [0] * len(unlabeled_test)})
        evaluate, _ = build_transforms()
        features, _ = extract_features(demo_frame, model_backbone, evaluate, device, args, "Dự đoán test không nhãn")
        probabilities = bundle["model"].predict_proba(scaler.transform(features))
        top = probabilities.argmax(axis=1)
        pd.DataFrame({
            "path": [str(p) for p in unlabeled_test],
            "predicted_class": [class_names[int(i)] for i in top],
            "confidence": probabilities.max(axis=1),
        }).to_csv(OUTPUT_DIR / "du_doan_test_khong_nhan.csv", index=False, encoding="utf-8-sig")

    print(f"\nĐã lưu biểu đồ, báo cáo, cache và mô hình tại: {OUTPUT_DIR}")


def main():
    args = parse_args()
    if args.augment < 0:
        raise ValueError("--augment phải >= 0")
    set_seed()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    frame, train, test, unlabeled_test, _, class_names = discover_dataset(args.data)
    make_eda(frame, train, class_names)
    arrays, backbone = load_or_extract(args, train, test, device)
    scaler, models_to_score, results, predictions, y_test, _ = fit_models(arrays)
    save_results(class_names, models_to_score, scaler, results, predictions, y_test,
                 backbone, unlabeled_test, device, args)


if __name__ == "__main__":
    main()
