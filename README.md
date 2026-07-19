# Vietnamese Traffic Sign Detection & Recognition

So sánh 4 hướng tiếp cận trên bộ dữ liệu biển báo giao thông Việt Nam
([maitam/vietnamese-traffic-signs](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs) trên Kaggle):

**Phân loại (classification)** — trên các crop biển báo đã có ground-truth bbox
| Model | Ghi chú |
|---|---|
| HOG features + Linear SVM | Cổ điển, không cần GPU |
| SimpleCNN | Deep, training curve log lên WandB |

**Phát hiện (detection)** — trên toàn ảnh
| Model | Ghi chú |
|---|---|
| HSV color threshold + contour | Không cần train |
| YOLOv8n (fine-tuned) | Ultralytics, log lên WandB |

Cuối pipeline có demo end-to-end: ảnh thô → detect (YOLO) → crop → classify (CNN) → kết quả.

## Cấu trúc project

```
.
├── main.py                    # Chạy toàn bộ pipeline
├── requirements.txt
├── notebooks/
│   └── cpv-vlsp.ipynb         # Notebook gốc (Kaggle)
└── src/
    ├── config.py               # Đường dẫn dữ liệu / output (qua biến môi trường)
    ├── data_utils.py           # Load classes, split files, crop theo bbox
    ├── classify_hog_svm.py     # Nhánh 1: HOG + SVM
    ├── classify_cnn.py         # Nhánh 2: SimpleCNN (dataset, model, train/eval)
    ├── detect_classical.py     # Nhánh 3: HSV + contour detector
    ├── detect_yolo.py          # Nhánh 4: YOLOv8n train/eval
    ├── eda.py                  # Phân bố lớp
    └── pipeline_demo.py        # Demo end-to-end detect + classify
```

## Dữ liệu

Bộ dữ liệu cần có cấu trúc:

```
<DATA_BASE>/
    classes.txt
    images/
    labels/                  # YOLO-format .txt
    split_dataset/
        train_files.txt
        test_files.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
export DATA_BASE=/path/to/vietnamese-traffic-signs/archive
export WORK_DIR=/path/to/working/dir

# Full pipeline (cần GPU cho CNN/YOLO nhanh hơn, tùy chọn WandB logging)
python main.py --wandb

# Bỏ qua train/eval YOLO (ví dụ không có GPU)
python main.py --skip-yolo-train
```

Notebook gốc (chạy trên Kaggle, dùng `kaggle_secrets` cho WandB API key) được giữ lại ở
`notebooks/cpv-vlsp.ipynb` để tham khảo.

## Kết quả

Các chỉ số accuracy / macro-F1 (classification) và precision / recall / mAP (detection)
được in ra console và log lên WandB (nếu bật `--wandb`) khi chạy `main.py`.
