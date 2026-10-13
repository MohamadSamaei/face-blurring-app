from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from PIL import Image
import io
import sys
from contextlib import asynccontextmanager
from src.inference import InferenceService
from app import build_ui
import gradio as gr

ml_models = {}

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Loading AI models...")
    try:
        inference_object = InferenceService()
        ml_models["inference"] = inference_object
        print("Model loaded successfully")
    except Exception as e:
        print(f"Critical error. Could not load the model. {e}")
        sys.exit(1)

    yield

    print("Cleaning up resources...")        
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return ({"status": "ok"})

@app.post("/predict")
def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code = 415, detail = "File format should be either jpeg or png")
    image_bytes = file.file.read()
    MAX_FILE_SIZE = 5 * (2 ** 20)                                               # 5 MB
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code = 413, detail = "File too large. Max size is 5 MB.")
    image_pil = Image.open(io.BytesIO(image_bytes))
    prepared_image, x, y = ml_models["inference"].prepare_image(image_pil)
    (blurred_image, w), z = ml_models["inference"].predict(prepared_image)
    output_buffer = io.BytesIO()
    blurred_image.save(output_buffer, format = "jpeg")
    blurred_image_bytes = output_buffer.getvalue()

    return Response(content = blurred_image_bytes, media_type="image/jpeg")

def gradio_predict_wrapper(image_pil):
    return ml_models["inference"].predict(image_pil)

demo_app = build_ui(gradio_predict_wrapper)

app = gr.mount_gradio_app(app, demo_app, path = "/")






    

