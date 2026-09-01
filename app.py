"""
Gradio frontend for the Face Blurring project.

- Loads trained backbone + classification + regression heads.
- Accepts an image upload.
- Predicts category (human/cat/dog) and, for humans, draws a face bounding box.
"""

import os
import math
import torch
import gradio as gr
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.models import BackboneNN, ClassificationNN, RegressionNN


# Configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224

class_names = {
    0: "human",
    1: "cat",
    2: "dog",
}

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)

# Load models

backbone = BackboneNN(in_channels=3).to(device)
classification_head = ClassificationNN(in_features=512, num_classes=3).to(device)
regression_head = RegressionNN(in_channels=512).to(device)

# Update these paths if your weight filenames change.
backbone.load_state_dict(torch.load("weights/backbone.pth", map_location=device))
classification_head.load_state_dict(torch.load("weights/classification_head.pth", map_location=device))
regression_head.load_state_dict(torch.load("weights/regression_head.pth", map_location=device))

backbone.eval()
classification_head.eval()
regression_head.eval()


# Preprocessing

def prepare_image(image: Image.Image):
    """
    Preprocess an uploaded PIL image to match training preprocessing.

    - Resize (longest side -> 224).
    - Pad to 224x224 with zeros.
    - Normalize and convert to tensor.

    Returns:
        display_image: PIL image after resize/pad (for visualization).
        tensor: [1, 3, 224, 224] tensor ready for the model.
        meta: dict with scale and padding info (not currently used in prediction).
    """
    image = image.convert("RGB")
    width_image, height_image = image.size

    scale = min(image_size / width_image, image_size / height_image)
    width_scaled = int(width_image * scale)
    height_scaled = int(height_image * scale)

    if width_scaled != image_size:
        width_margin = image_size - width_scaled
        pad_left = round(width_margin / 2)
        pad_right = width_margin - pad_left
    else:
        pad_left, pad_right = 0, 0

    if height_scaled != image_size:
        height_margin = image_size - height_scaled
        pad_top = round(height_margin / 2)
        pad_bottom = height_margin - pad_top
    else:
        pad_top, pad_bottom = 0, 0

    image = TF.resize(image, size=[height_scaled, width_scaled])
    image = TF.pad(
        image,
        padding=[pad_left, pad_top, pad_right, pad_bottom],
        fill=0,
        padding_mode="constant",
    )

    tensor = transform(image).unsqueeze(0).to(device)

    meta = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "new_w": width_scaled,
        "new_h": height_scaled,
    }

    return image, tensor, meta


# Prediction

@torch.no_grad()
def predict(image):
    """
    Predict category and (if human) face bounding box for an uploaded image.

    Returns:
        (display_image, annotations), summary_text
    """
    if image is None:
        return None, "No image uploaded."

    display_image, image_tensor, meta = prepare_image(image)

    features = backbone(image_tensor)
    class_logits = classification_head(features)
    box_pred = regression_head(features)

    class_index = torch.argmax(class_logits, dim=1).item()
    predicted_label = class_names[class_index]

    box = box_pred.squeeze(0)  # [4]

    x1 = int(box[0].item() * image_size)
    y1 = int(box[1].item() * image_size)
    w = int(box[2].item() * image_size)
    h = int(box[3].item() * image_size)

    x2 = x1 + w
    y2 = y1 + h

    # Ensure box is ordered for AnnotatedImage.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Blur the predicted face region (only for humans)
    if predicted_label == "human" and x2 > x1 and y2 > y1:
        face_crop = display_image.crop((x1, y1, x2, y2))
        blurred_face = face_crop.filter(ImageFilter.GaussianBlur(radius=10))
        display_image.paste(blurred_face, (x1, y1))


    annotations = []
    if predicted_label == "human":
        annotations.append(((x1, y1, x2, y2), "face"))

    if class_index == 0:
        summary = (
            f"Predicted category: {predicted_label}\n"
            f"Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}"
        )
    else:
        summary = (
            f"Predicted category: {predicted_label}\n"
            "No human face detected."
        )

    return (display_image, annotations), summary



# Gradio interface

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload photo"),
    outputs=[
        gr.AnnotatedImage(
            label="Prediction",
            color_map={"face": "#ff4d4f"},
            height=400,
        ),
        gr.Textbox(label="Result"),
    ],
    title="Face Bounding Box Demo",
    description=(
        "Upload a photo. The app predicts whether the image is human, cat, or dog. "
        "For human images, a bounding box is drawn around the predicted face location."
    ),
)


if __name__ == "__main__":
    demo.launch()