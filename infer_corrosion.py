#!/usr/bin/env python3
"""
Batch inference utility for the corrosion classifier.

Example:
    python infer_corrosion.py --weights resnet_corrosion.pt --input-folder new_images --device cpu
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from torchvision import transforms

from train_resnet_classifier import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrosion classifier on a folder of images.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights (.pt file).")
    parser.add_argument("--input-folder", type=Path, required=True, help="Folder containing images to classify.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV file to store predictions.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize shorter side to this size.")
    parser.add_argument("--classes", nargs="+", default=["no_corrosion", "corrosion"], help="Class names.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary outputs.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout used when constructing the model head.")
    parser.add_argument("--use-mobilenet", action="store_true", help="Use MobileNetV3-Large backbone.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    return parser.parse_args()


def load_weights(args: argparse.Namespace, classes: Sequence[str]) -> torch.nn.Module:
    checkpoint = torch.load(args.weights, map_location=args.device)
    ckpt_classes = checkpoint.get("classes")
    if ckpt_classes:
        classes = ckpt_classes
        print(f"Loaded class names from checkpoint: {classes}")
    binary = len(classes) == 2
    model = build_model(
        num_classes=len(classes), dropout=args.dropout, use_mobilenet=args.use_mobilenet, binary=binary
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()
    return model, classes, binary


def main() -> None:
    args = parse_args()
    if not args.input_folder.exists():
        raise SystemExit(f"Input folder not found: {args.input_folder}")

    model, classes, binary = load_weights(args, args.classes)
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    results: List[dict] = []
    image_paths = sorted(
        p for p in args.input_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        raise SystemExit(f"No supported image files found in {args.input_folder}")

    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(args.device)
            logits = model(tensor)
            if binary:
                prob = torch.sigmoid(logits)[0, 0].item()
                pred_idx = 1 if prob >= args.threshold else 0
                confidence = prob if pred_idx == 1 else 1.0 - prob
            else:
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx_tensor = torch.max(probs, dim=1)
                confidence = confidence.item()
                pred_idx = pred_idx_tensor.item()
            result = {
                "image": str(img_path),
                "prediction": classes[pred_idx],
                "confidence": confidence,
            }
            results.append(result)
            print(f"{img_path.name}: {result['prediction']} ({confidence:.3f})")

    if args.output_csv:
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "prediction", "confidence"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
