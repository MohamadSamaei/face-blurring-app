"""
Entry point for the Face Blurring project.

Usage:
    python main.py --mode train
    python main.py --mode val
    python main.py --mode visualize
"""

import os
import argparse

import cv2
import yaml
import torch
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.models import BackboneNN, ClassificationNN, RegressionNN
from src.dataset_train import HumanCatDogClass_train
from src.dataset_val import HumanCatDogClass_val
from src.train import training_loop
from src.validate import validation_loop
from src.visualize import draw_bounding_box

# Class counts used to build the inverse-frequency sampler weights (training set only).
human_train_samples = 8000
cat_train_samples = 375
dog_train_samples = 457

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
image_size = 224


def load_config(config_path):
    """Load hyperparameters and paths from a YAML config file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_val_transform():
    """Deterministic preprocessing: resize + pad + normalize. No augmentation."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
            ),
            A.Normalize(mean=imagenet_mean, std=imagenet_std, max_pixel_value=255.0),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"],
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


def get_train_transform():
    """Training preprocessing: resize + pad + geometric/photometric augmentation + normalize."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
            ),
            A.HorizontalFlip(p=0.5),
            A.BBoxSafeRandomCrop(erosion_rate=0, p=0.5),
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_CUBIC),
            A.Affine(
                translate_percent=0.1,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                p=0.4,
            ),
            # Simulate poor photo quality.
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1),
                    A.MedianBlur(blur_limit=3, p=1),
                    A.GaussNoise(std_range=(0.0124, 0.0248), p=1),
                    A.ImageCompression(quality_range=(60, 90), p=1),
                ],
                p=0.3,
            ),
            # Simulate different lighting conditions.
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0),
                ],
                p=0.5,
            ),
            A.Normalize(mean=imagenet_mean, std=imagenet_std, max_pixel_value=255.0),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"],
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


def build_weighted_sampler(dataset):
    """Build a WeightedRandomSampler using inverse class frequency (human/cat/dog)."""
    class_weights = {
        0: 1 / human_train_samples,
        1: 1 / cat_train_samples,
        2: 1 / dog_train_samples,
    }
    category_list = dataset.category.tolist()
    sample_weights = [class_weights[category] for category in category_list]

    # replacement=True: without it, minority-class samples would run out quickly
    # since they're picked more often relative to their small count.
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def custom_collate_fn(batch):
    """Stack tensors normally, but keep image_ids as a plain list of strings."""
    images, coordinates, categories, image_ids = zip(*batch)
    images = torch.stack(images)
    coordinates = torch.stack(coordinates)
    categories = torch.stack(categories)
    return images, coordinates, categories, image_ids


def build_models(device):
    """Instantiate backbone + classification head + regression head."""
    backbone = BackboneNN(in_channels=3).to(device)
    classification_head = ClassificationNN(in_features=512, num_classes=3).to(device)
    regression_head = RegressionNN(in_channels=512).to(device)
    return backbone, classification_head, regression_head


def run_train(config, backbone, classification_head, regression_head, device):
    print("Loading training dataset...")
    train_ds = HumanCatDogClass_train(
        csv_address=config["data"]["train_csv"],
        folder_path=config["data"]["image_folder_train"],
        transform=get_train_transform(),
    )

    sampler_train = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,  # shuffle is handled by the sampler
        num_workers=config["training"]["num_workers"],
        collate_fn=custom_collate_fn,
        sampler=sampler_train,
    )

    print("Starting training loop...")
    training_loop(
        train_loader,
        backbone_model=backbone,
        classification_model=classification_head,
        regression_model=regression_head,
        config=config,
        device=device,
    )

    os.makedirs("weights", exist_ok=True)
    torch.save(backbone.state_dict(), "weights/backbone.pth")
    torch.save(classification_head.state_dict(), "weights/classification_head.pth")
    torch.save(regression_head.state_dict(), "weights/regression_head.pth")
    print("Models saved successfully in weights/ folder.")


def run_val(config, backbone, classification_head, regression_head, device):
    print("Loading saved weights...")
    if not os.path.exists("weights/backbone.pth"):
        raise FileNotFoundError("Could not find saved weights. Run --mode train first.")

    backbone.load_state_dict(torch.load("weights/backbone.pth", map_location=device))
    classification_head.load_state_dict(torch.load("weights/classification_head.pth", map_location=device))
    regression_head.load_state_dict(torch.load("weights/regression_head.pth", map_location=device))

    print("Loading validation dataset...")
    val_ds = HumanCatDogClass_val(
        csv_address=config["data"]["val_csv"],
        folder_path=config["data"]["image_folder_eval"],
        transform=get_val_transform(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn,
    )

    print("Starting validation loop...")
    validation_loop(
        val_loader,
        backbone_model=backbone,
        classification_model=classification_head,
        regression_model=regression_head,
        config=config,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Pipeline for the Face Blurring project")
    parser.add_argument("--mode", type=str, choices=["train", "val", "visualize"], required=True,
                         help="Which step to run")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                         help="Path to the config file")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "visualize":
        print("Running visualization...")
        draw_bounding_box(config)
        return

    print("Initializing models...")
    backbone, classification_head, regression_head = build_models(device)

    if args.mode == "train":
        run_train(config, backbone, classification_head, regression_head, device)
    elif args.mode == "val":
        run_val(config, backbone, classification_head, regression_head, device)


if __name__ == "__main__":
    main()
