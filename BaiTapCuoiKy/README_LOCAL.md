# Chạy project Oxford Flowers 102 trên Windows

## 1. Cài môi trường

Mở PowerShell tại thư mục project:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

PyTorch được cài riêng để tránh pip thay nhầm bản CUDA bằng bản CPU. Với RTX 4070 như máy hiện tại:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Sau đó cài các thư viện còn lại (bao gồm Streamlit cho giao diện web):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` không cài lại torch/torchvision; vì vậy chạy lệnh trên không thay bản CUDA. Máy chỉ dùng CPU thì chọn bản PyTorch CPU phù hợp trên [trang cài đặt chính thức](https://pytorch.org/get-started/locally/), rồi cài `requirements.txt`.

## 2. Cấu trúc dữ liệu

Mặc định chương trình tìm dữ liệu ở `dataset/flower_data`:

```text
dataset/flower_data/
  train/<mã_lớp>/*.jpg
  valid/<mã_lớp>/*.jpg
  test/*.jpg                 # ảnh không nhãn, chỉ dùng demo dự đoán
  cat_to_name.json
```

Nếu dataset nằm ở nơi khác, truyền đường dẫn khi huấn luyện:

```powershell
.\.venv\Scripts\python.exe nhom19.py --data "D:\duong-dan\flower_data"
```

## 3. Huấn luyện và đánh giá

```powershell
.\.venv\Scripts\python.exe nhom19.py
```

Chương trình tự chọn CUDA khi khả dụng. Lần đầu tải trọng số ImageNet và trích xuất ảnh; cache được lưu trong `outputs/cache/`. Các lần sau dùng lại cache. Mặc định dùng ResNet18, không thêm lượt augmentation để tiết kiệm thời gian.

Để dùng ResNet50 hoặc augmentation:

```powershell
.\.venv\Scripts\python.exe nhom19.py --backbone resnet50 --augment 1
```

Muốn trích xuất lại đặc trưng (ví dụ sau khi đổi từ CPU sang GPU):

```powershell
.\.venv\Scripts\python.exe nhom19.py --force-features
```

`train/` được dùng để huấn luyện và chọn K bằng Stratified 5-fold cross-validation; `valid/` được giữ nguyên để đánh giá cuối. `test/` không có nhãn nên chỉ dùng chạy demo dự đoán. Đồ thị và bảng điểm chọn K được lưu tại `outputs/knn_chon_k_cv.png` và `outputs/knn_chon_k_cv.csv`.

## 4. Mở giao diện web

Trước tiên cần có model `outputs/flower_classifier.joblib`, được tạo sau khi chạy `nhom19.py`. Sau đó mở giao diện:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Trình duyệt sẽ mở trang local. Tải một hoặc nhiều ảnh JPG/PNG/WEBP để nhận diện Top‑5; tab **Đánh giá mô hình** hiển thị bảng metric, biểu đồ so sánh, confusion matrix và phân bố dữ liệu.

## 5. Kết quả đầu ra

Các biểu đồ EDA, metric, báo cáo từng lớp, confusion matrix, model và dự đoán ảnh test không nhãn được lưu trong `outputs/`.
