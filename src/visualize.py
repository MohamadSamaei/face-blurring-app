"""
Visualization script for the Face Blurring project.

Reads predicted and ground-truth boxes from the validation CSV,
resizes/pads each image to match the model's preprocessing,
draws both GT and predicted bounding boxes, and blurs the predicted face region.
"""

import csv
import os
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms.functional import resize, pad

image_size = 224


def resize_and_pad_image(image, target_size=image_size):
    """
    Resize + pad an image to target_size x target_size, matching training preprocessing.

    - LongestMaxSize-style resize (scale so longest side == target_size).
    - Pad with zeros to reach target_size x target_size.
    """
    width_image, height_image = image.size
    scale = min(target_size / width_image, target_size / height_image)

    width_scaled = int(width_image * scale)
    height_scaled = int(height_image * scale)

    image = resize(image, size=[height_scaled, width_scaled])

    if width_scaled != target_size:
        width_margin = target_size - width_scaled
        pad_left = round(width_margin / 2)
        pad_right = width_margin - pad_left
    else:
        pad_left, pad_right = 0, 0

    if height_scaled != target_size:
        height_margin = target_size - height_scaled
        pad_top = round(height_margin / 2)
        pad_bottom = height_margin - pad_top
    else:
        pad_top, pad_bottom = 0, 0

    image = pad(
        image,
        padding=[pad_left, pad_top, pad_right, pad_bottom],
        fill=0,
        padding_mode="constant",
    )
    return image


def draw_bounding_box(config):
    """
    Draw GT and predicted bounding boxes on validation images, and blur the predicted face region.

    - Reads predictions from the CSV specified in config.
    - Saves annotated images to the output folder in config.
    - Skips images that are not found on disk.
    """

    csv_path = config["data"]["output_csv"]
    output_folder = config["data"]["output_bb_folder"]
    images_folder = config["data"]["image_folder_eval"]
    target_size = config["model"]["width_target"]  # assumed square
    box_color_pred = config["bb"]["box_color_pred"]
    box_color_gt = config["bb"]["box_color_gt"]
    box_width = config["bb"]["box_width"]

    os.makedirs(output_folder, exist_ok=True)

    with open(csv_path, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            image_id = row["image_id"]

            # Predicted boxes
            x1_pred = int(float(row["x_1_pred"]))
            y1_pred = int(float(row["y_1_pred"]))
            w_pred = int(float(row["width_pred"]))
            h_pred = int(float(row["height_pred"]))

            x2_pred = x1_pred + w_pred
            y2_pred = y1_pred + h_pred

            # Ground-truth boxes
            x1_gt = int(float(row["x_1"]))
            y1_gt = int(float(row["y_1"]))
            w_gt = int(float(row["width"]))
            h_gt = int(float(row["height"]))

            x2_gt = x1_gt + w_gt
            y2_gt = y1_gt + h_gt

            image_path = os.path.join(images_folder, image_id)
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            image = resize_and_pad_image(image, target_size=target_size)

            # Ensure coordinates are ordered (x1 <= x2, y1 <= y2)
            pred_box = [
                min(x1_pred, x2_pred),
                min(y1_pred, y2_pred),
                max(x1_pred, x2_pred),
                max(y1_pred, y2_pred),
            ]
            gt_box = [
                min(x1_gt, x2_gt),
                min(y1_gt, y2_gt),
                max(x1_gt, x2_gt),
                max(y1_gt, y2_gt),
            ]

            # Blur the predicted face region
            x1b, y1b, x2b, y2b = pred_box
            if x2b > x1b and y2b > y1b:
                face_crop = image.crop((x1b, y1b, x2b, y2b))
                blurred_face = face_crop.filter(ImageFilter.GaussianBlur(radius=10))
                image.paste(blurred_face, (x1b, y1b))

            draw = ImageDraw.Draw(image)

            # Draw GT box (green) and predicted box (red)
            draw.rectangle(gt_box, outline=box_color_gt, width=box_width)
            draw.rectangle(pred_box, outline=box_color_pred, width=box_width)

            output_path = os.path.join(output_folder, image_id)
            image.save(output_path)

    print(f"Images with bounding box are at: {output_folder}")