"""
Vietnamese Traffic Sign Detection & Recognition - full pipeline runner.

So sanh 4 huong tiep can tren cung mot bo du lieu VLSP bien bao giao thong VN:
  - Phan loai (classification, tren crop bien bao da co ground-truth bbox):
      1) HOG features + Linear SVM
      2) SimpleCNN
  - Phat hien (detection, tren toan anh):
      3) HSV color threshold + contour (khong can train)
      4) YOLOv8n fine-tuned

Usage:
    export DATA_BASE=/path/to/vietnamese-traffic-signs/archive
    export WORK_DIR=/path/to/working/dir
    python main.py --skip-yolo-train      # eval bang checkpoint co san
    python main.py                        # chay full pipeline (can GPU + WandB key)
"""
import argparse
import os

import torch

from src import config, data_utils, eda
from src.classify_cnn import SimpleCNN, build_dataloaders, evaluate_cnn, train_cnn
from src.classify_hog_svm import train_and_eval_svm
from src.detect_classical import evaluate_classical_detector
from src.detect_yolo import build_yolo_dataset_files, eval_yolo, train_yolo
from src.pipeline_demo import full_pipeline_demo


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs-cnn", type=int, default=15)
    p.add_argument("--epochs-yolo", type=int, default=30)
    p.add_argument("--skip-yolo-train", action="store_true", help="Skip YOLO training/eval (e.g. no GPU)")
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-api-key", type=str, default=os.environ.get("WANDB_API_KEY"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- Load data -------------------------------------------------------
    classes = data_utils.load_classes()
    train_files, test_files = data_utils.load_file_lists()
    print("So lop:", len(classes), "| Train:", len(train_files), "| Test:", len(test_files))

    X_train_img, y_train = data_utils.load_split_crops(train_files)
    X_test_img, y_test = data_utils.load_split_crops(test_files)
    y_train_idx, y_test_idx, unique_classes = data_utils.remap_labels(y_train, y_test)
    n_classes = len(unique_classes)
    print("n_classes thuc te dung de train:", n_classes)

    wandb_run = None
    if args.wandb:
        import wandb

        if args.wandb_api_key:
            wandb.login(key=args.wandb_api_key)
        wandb_run = wandb.init(project=config.WANDB_PROJECT_CLASSIFICATION, name="simplecnn-vs-svm")

    # ---- Branch 1: HOG + SVM ---------------------------------------------
    _, acc_svm, f1_svm = train_and_eval_svm(X_train_img, y_train_idx, X_test_img, y_test_idx)

    # ---- Branch 2: CNN ------------------------------------------------------
    train_loader, test_loader = build_dataloaders(X_train_img, y_train_idx, X_test_img, y_test_idx)
    model_cnn = SimpleCNN(n_classes).to(device)
    model_cnn = train_cnn(model_cnn, train_loader, test_loader, device, epochs=args.epochs_cnn, wandb_run=wandb_run)
    acc_cnn, f1_cnn = evaluate_cnn(model_cnn, test_loader, device)

    if wandb_run is not None:
        wandb_run.log(
            {"final/svm_acc": acc_svm, "final/svm_f1_macro": f1_svm, "final/cnn_acc": acc_cnn, "final/cnn_f1_macro": f1_cnn}
        )
        wandb_run.finish()

    print("\n=== So sanh co dien vs deep (Test set) ===")
    print(f"SVM (HOG features): accuracy={acc_svm:.4f} | macro-F1={f1_svm:.4f}")
    print(f"CNN (deep):          accuracy={acc_cnn:.4f} | macro-F1={f1_cnn:.4f}")

    # ---- Branch 3: classical HSV + contour detector -----------------------
    precision_cls, recall_cls, f1_cls = evaluate_classical_detector(test_files)

    # ---- Branch 4: YOLOv8n detector ---------------------------------------
    metrics_yolo = None
    if not args.skip_yolo_train:
        data_yaml_path = build_yolo_dataset_files(classes, train_files, test_files)
        model_yolo, _ = train_yolo(data_yaml_path, epochs=args.epochs_yolo, wandb_api_key=args.wandb_api_key)
        metrics_yolo = eval_yolo(model_yolo, data_yaml_path)

        # ---- EDA + end-to-end demo -----------------------------------------
        eda.plot_class_distribution(y_train, y_test, classes)
        for fname in test_files[:3]:
            print(f"\n--- {fname} ---")
            full_pipeline_demo(f"{config.BASE}/images/{fname}", model_yolo, model_cnn, classes, unique_classes, device)

    # ---- Final summary ------------------------------------------------------
    print("\n=== TONG HOP 4 MODEL ===")
    print(f"Detection - Classical (HSV+contour): P={precision_cls:.4f} R={recall_cls:.4f} F1={f1_cls:.4f}")
    if metrics_yolo is not None:
        print(
            f"Detection - YOLOv8n (CNN):           mAP50={metrics_yolo.box.map50:.4f} "
            f"P={metrics_yolo.box.mp:.4f} R={metrics_yolo.box.mr:.4f}"
        )
    print(f"Classification - SVM (HOG):          acc={acc_svm:.4f} F1={f1_svm:.4f}")
    print(f"Classification - CNN:                acc={acc_cnn:.4f} F1={f1_cnn:.4f}")


if __name__ == "__main__":
    main()
