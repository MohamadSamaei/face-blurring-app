import torch
from torchvision import transforms
from src.models import BackboneNN, ClassificationNN, RegressionNN
import PIL
from PIL import ImageFilter
import torchvision.transforms.functional as TF

class InferenceService:
    image_size = 224
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    class_names = {
    0: "human",
    1: "cat",
    2: "dog",
}
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.imagenet_mean, self.imagenet_std)
        ])
        self.backbone_weights = "weights/backbone.pth"
        self.classification_weights = "weights/classification_head.pth"
        self.regression_weights = "weights/regression_head.pth"
        self.backbone = BackboneNN(in_channels=3).to(self.device)
        self.classification_head = ClassificationNN(in_features = 512, num_classes=3).to(self.device)
        self.regression_head = RegressionNN(in_channels=512).to(self.device)
        self.backbone.load_state_dict(torch.load(self.backbone_weights, map_location=self.device))
        self.classification_head.load_state_dict(torch.load(self.classification_weights, map_location=self.device))
        self.regression_head.load_state_dict(torch.load(self.regression_weights, map_location=self.device))
        self.backbone.eval()
        self.classification_head.eval()
        self.regression_head.eval()

    def prepare_image(self, image: PIL.Image):                      #pre-processing
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

        scale = min(InferenceService.image_size / width_image, InferenceService.image_size / height_image)
        width_scaled = int(width_image * scale)
        height_scaled = int(height_image * scale)

        if width_scaled != InferenceService.image_size:
            width_margin = InferenceService.image_size - width_scaled
            pad_left = round(width_margin / 2)
            pad_right = width_margin - pad_left
        else:
            pad_left, pad_right = 0, 0

        if height_scaled != InferenceService.image_size:
            height_margin = InferenceService.image_size - height_scaled
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

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        meta = {
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "new_w": width_scaled,
            "new_h": height_scaled,
        }

        return image, tensor, meta    

    @torch.no_grad()
    def predict(self, image):
        """
        Predict category and (if human) face bounding box for an uploaded image.
    
        Returns:
            (display_image, annotations), summary_text
        """
        if image is None:
            return None, "No image uploaded."
    
        display_image, image_tensor, meta = self.prepare_image(image)
    
        features = self.backbone(image_tensor)
        class_logits = self.classification_head(features)
        box_pred = self.regression_head(features)
    
        class_index = torch.argmax(class_logits, dim=1).item()
        predicted_label = InferenceService.class_names[class_index]
    
        box = box_pred.squeeze(0)  # [4]
    
        x1 = int(box[0].item() * InferenceService.image_size)
        y1 = int(box[1].item() * InferenceService.image_size)
        w = int(box[2].item() * InferenceService.image_size)
        h = int(box[3].item() * InferenceService.image_size)
    
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
        







        
