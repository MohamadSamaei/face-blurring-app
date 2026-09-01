"""
Training dataset for the Face Blurring project.

Applies Albumentations transforms (resize, pad, augmentation, normalize) and updates
bounding boxes in COCO format. Returns (image_tensor, coords_norm, category, image_id).
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
import cv2

image_size = 224
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


class HumanCatDogClass_train(Dataset):
    """
    Dataset for training with Albumentations.

    Expected CSV columns: image_id, category, x_1, y_1, width, height
    - category: 0 = human (has bbox), 1 = cat, 2 = dog (no bbox)
    - bbox in COCO format [x_min, y_min, width, height] in pixels.
    """

    def __init__(self, csv_address, folder_path, transform=None, image_size=image_size):
        self.csv = pd.read_csv(csv_address)
        self.csv = self.csv.dropna(subset=["image_id", "category"]).reset_index(drop=True)

        self.files_folder = folder_path
        self.image_id = self.csv["image_id"].astype(str)
        self.coordinates = self.csv[["x_1", "y_1", "width", "height"]].values.astype(np.float32)
        self.category = self.csv["category"].values
        self.image_size = image_size

        # Default transform if none is provided. This is a only in case that in the main.py the transforms are not provided.
        self.transform = transform if transform is not None else A.Compose(
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
                min_visibility=0.3,
            ),
        )

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        image_id = self.image_id.iloc[index]
        bbox = self.coordinates[index].copy()
        category = self.category[index]

        # Robust category parsing (handle NaN / non-numeric)
        try:
            category_float = float(category)
            if np.isnan(category_float):
                category = 0
            else:
                category = int(category_float)
        except (ValueError, TypeError):
            category = 0

        category_tensor = torch.tensor(category, dtype=torch.long)

        image_path = os.path.join(self.files_folder, image_id)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)  # HWC, RGB

        # Humans have a face bbox; cats/dogs do not.
        if category == 0:
            bboxes = [bbox.tolist()]
            class_labels = [category]
        else:
            bboxes = []
            class_labels = []

        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
        )

        image = transformed["image"]
        # Albumentations returns HWC; convert to CHW tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Rebuild normalized coordinates from transformed bbox.
        if category == 0 and len(transformed["bboxes"]) > 0:
            x_1, y_1, width_box, height_box = transformed["bboxes"][0]

            x_1_norm = x_1 / image_size
            y_1_norm = y_1 / image_size
            width_box_norm = width_box / image_size
            height_box_norm = height_box / image_size

            coordinates = torch.tensor(
                [x_1_norm, y_1_norm, width_box_norm, height_box_norm],
                dtype=torch.float32,
            )
        else:
            coordinates = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        return image, coordinates, category_tensor, image_id