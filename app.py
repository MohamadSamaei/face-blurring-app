"""
Gradio frontend for the Face Blurring project.

- Loads trained backbone + classification + regression heads.
- Accepts an image upload.
- Predicts category (human/cat/dog) and, for humans, draws a face bounding box.
"""

import gradio as gr
from src.inference import InferenceService

def build_ui(predict_function):
    demo = gr.Interface(
        fn=predict_function,
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

    return demo    