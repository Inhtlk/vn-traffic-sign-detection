"""
YOLOv8n (CNN detector): chuan bi data.yaml, train, va eval mAP.
"""
import os
from typing import List

from . import config


def build_yolo_dataset_files(
    classes: List[str],
    train_files: List[str],
    test_files: List[str],
    base: str = config.BASE,
    work: str = config.WORK,
) -> str:
    """
    Write train.txt / val.txt / data.yaml pointing directly at the raw image
    folder, in the format Ultralytics expects. Returns the path to data.yaml.
    """
    os.makedirs(work, exist_ok=True)

    with open(f"{work}/train.txt", "w") as f:
        for fname in train_files:
            f.write(f"{base}/images/{fname}\n")

    with open(f"{work}/val.txt", "w") as f:
        for fname in test_files:
            f.write(f"{base}/images/{fname}\n")

    yaml_content = f"""
path: {work}
train: train.txt
val: val.txt
nc: {len(classes)}
names: {classes}
"""
    data_yaml_path = f"{work}/data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(yaml_content[:300])
    return data_yaml_path


def train_yolo(data_yaml_path: str, epochs: int = 30, imgsz: int = 640, batch: int = 16, wandb_api_key: str = None):
    """Fine-tune a pretrained yolov8n.pt on the traffic-sign dataset."""
    from ultralytics import YOLO

    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    model_yolo = YOLO("yolov8n.pt")  # pretrained nho, fine-tune tiep tren data VN
    results = model_yolo.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=config.WANDB_PROJECT_DETECTION,
        name="yolov8n-finetune",
    )
    return model_yolo, results


def eval_yolo(model_yolo, data_yaml_path: str):
    """Run validation and print mAP50 / mAP50-95 / precision / recall."""
    metrics = model_yolo.val(data=data_yaml_path, split="val")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    return metrics
