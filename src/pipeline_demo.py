"""
Pipeline demo end-to-end: anh tho -> detect (YOLO) -> crop -> classify (CNN) -> ket qua.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config


def full_pipeline_demo(
    img_path: str,
    model_yolo,
    model_cnn,
    classes,
    unique_classes,
    device,
    conf_thres: float = 0.25,
    save_path: str = None,
):
    """Run detection + classification on a single image and visualize the result."""
    save_path = save_path or f"{config.WORK}/demo_result.png"

    results = model_yolo.predict(img_path, conf=conf_thres, verbose=False)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    boxes = results[0].boxes
    if len(boxes) == 0:
        print("Khong phat hien bien bao nao.")
        return

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        crop = img[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (64, 64)).astype(np.float32) / 255.0
        crop_tensor = torch.tensor(crop_resized).permute(2, 0, 1).unsqueeze(0).to(device)

        model_cnn.eval()
        with torch.no_grad():
            pred_idx = model_cnn(crop_tensor).argmax(1).item()
        pred_class = classes[unique_classes[pred_idx]]

        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, y1 - 5, pred_class, color="white", fontsize=10, backgroundcolor="green")
        print(f"Phat hien bbox ({x1},{y1},{x2},{y2}) -> phan loai: {pred_class}")

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
