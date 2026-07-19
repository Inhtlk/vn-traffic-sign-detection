"""
Nhanh 2: CNN classifier (deep, co training curve log len WandB).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score as sk_f1
from torch.utils.data import DataLoader, Dataset


class SignDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # HWC -> CHW
        return img, self.labels[idx]


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64->32
        x = self.pool(F.relu(self.conv2(x)))  # 32->16
        x = self.pool(F.relu(self.conv3(x)))  # 16->8
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def build_dataloaders(X_train_img, y_train_idx, X_test_img, y_test_idx, batch_size=64):
    train_ds = SignDataset(X_train_img, y_train_idx)
    test_ds = SignDataset(X_test_img, y_test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_cnn(model, train_loader, test_loader, device, epochs=15, lr=1e-3, wandb_run=None):
    """Train SimpleCNN, optionally logging metrics to an active wandb run."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                correct += (out.argmax(1) == labels).sum().item()
                total += imgs.size(0)
        test_acc = correct / total

        if wandb_run is not None:
            wandb_run.log(
                {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}
            )
        print(
            f"epoch {epoch + 1}/{epochs} | train_loss {train_loss:.4f} "
            f"| train_acc {train_acc:.4f} | test_acc {test_acc:.4f}"
        )

    return model


def evaluate_cnn(model, test_loader, device):
    """Compute accuracy and macro-F1 for a trained CNN on the test set."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc_cnn = accuracy_score(all_labels, all_preds)
    f1_cnn = sk_f1(all_labels, all_preds, average="macro")
    return acc_cnn, f1_cnn
