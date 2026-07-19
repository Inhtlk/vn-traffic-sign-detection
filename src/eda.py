"""
EDA nhanh: phan bo so luong anh theo tung lop bien bao.
"""
from collections import Counter
from typing import List

import matplotlib.pyplot as plt

from . import config


def plot_class_distribution(y_train, y_test, classes: List[str], save_path: str = None):
    """Bar chart of sample counts per class, plus imbalance summary."""
    save_path = save_path or f"{config.WORK}/class_distribution.png"

    all_labels = list(y_train) + list(y_test)
    class_counts = Counter(all_labels)

    plt.figure(figsize=(14, 4))
    plt.bar(
        [classes[c] for c in sorted(class_counts)],
        [class_counts[c] for c in sorted(class_counts)],
    )
    plt.xticks(rotation=90)
    plt.title(f"Phan bo {len(class_counts)} lop bien bao trong dataset")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    most_common = max(class_counts.items(), key=lambda x: x[1])
    least_common = min(class_counts.items(), key=lambda x: x[1])
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())

    print("Lop nhieu nhat:", most_common)
    print("Lop it nhat:", least_common)
    print("Mat can bang (max/min ratio):", imbalance_ratio)

    return class_counts
