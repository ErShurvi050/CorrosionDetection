#!/usr/bin/env python3
"""
Fine-tune a pretrained ResNet on the corrosion classification task built from dataset2_yolo.

Usage (after installing torch/torchvision):
    python train_resnet_classifier.py --data-root dataset2_yolo --epochs 25
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


class CorrosionImageDataset(Dataset):
    """Wrap YOLO split to act as a classification dataset (label = corrosion present?)."""

    def __init__(self, image_dir: Path, label_dir: Path, transform: transforms.Compose):
        self.samples: List[Tuple[Path, int]] = []
        for img_path in sorted(image_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            label_path = label_dir / f"{img_path.stem}.txt"
            label = 1 if label_path.exists() and label_path.read_text().strip() else 0
            self.samples.append((img_path, label))
        if not self.samples:
            raise ValueError(f"No images discovered in {image_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label


def build_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    splits = {}
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    for split in ("train", "val", "test"):
        tfms = train_tfms if split == "train" else eval_tfms
        splits[split] = CorrosionImageDataset(
            image_dir=data_root / "images" / split,
            label_dir=data_root / "labels" / split,
            transform=tfms,
        )
    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers)
        for name, ds in splits.items()
    }
    return loaders


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    binary: bool,
) -> float:
    model.train()
    running = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if binary:
            labels = labels.float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * inputs.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    correct = 0
    tp = fp = fn = tn = 0
    binary = isinstance(criterion, nn.BCEWithLogitsLoss)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        targets = labels
        if binary:
            targets = targets.float().unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        if binary:
            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs >= 0.5).long()
            correct += (preds == labels).sum().item()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
        else:
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    loss_avg = running_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    metrics = {"accuracy": acc}
    if binary:
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        denom = precision + recall + 1e-8
        f1 = 2 * precision * recall / denom
        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return loss_avg, acc, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 for corrosion classification.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset2_yolo"), help="Root of YOLOv8 dataset.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save-path", type=Path, default=Path("resnet_corrosion.pt"))
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["no_corrosion", "corrosion"],
        help="Class names (order defines label indices).",
    )
    parser.add_argument("--use-mobilenet", action="store_true", help="Swap backbone to MobileNetV3-Large.")
    return parser.parse_args()


def build_model(num_classes: int, dropout: float, use_mobilenet: bool, binary: bool) -> nn.Module:
    output_dim = 1 if binary else num_classes
    if use_mobilenet:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, output_dim)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, output_dim))
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    binary = len(args.classes) == 2
    loaders = build_dataloaders(args.data_root, args.image_size, args.batch_size, args.num_workers)
    model = build_model(
        num_classes=len(args.classes), dropout=args.dropout, use_mobilenet=args.use_mobilenet, binary=binary
    ).to(device)
    criterion: nn.Module = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], criterion, optimizer, device, binary=binary)
        val_loss, val_acc, val_metrics = evaluate(model, loaders["val"], criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | Train loss {train_loss:.4f} "
            f"| Val loss {val_loss:.4f} | Val acc {val_acc:.3f}"
        )
        if binary:
            print(
                f"  Precision {val_metrics['precision']:.3f} "
                f"| Recall {val_metrics['recall']:.3f} | F1 {val_metrics['f1']:.3f}"
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "val_acc": val_acc, "classes": args.classes},
                args.save_path,
            )
            print(f"Saved new best weights to {args.save_path}")

    test_loss, test_acc, test_metrics = evaluate(model, loaders["test"], criterion, device)
    print(f"Test loss {test_loss:.4f} | Test acc {test_acc:.3f}")
    if binary:
        print(
            f"Test precision {test_metrics['precision']:.3f} "
            f"| Recall {test_metrics['recall']:.3f} | F1 {test_metrics['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
