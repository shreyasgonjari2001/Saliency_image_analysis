from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import numpy as np
import torch
import easyocr
from ultralytics import YOLO
from utils import load_model, gaussian_centerbias, gini_index_heatmap, attention_entropy, average_color_saturation, saliency_weighted_vividness, color_temperature
from utils_obj_detection import get_obj_bboxes_labels, attention_share_per_object, gini_index_across_objects, saliency_object_alignment_index, persons_size_ratio
from ocr import get_text_data, get_txt_impact_score, relative_font_size_metric, text_size_variability, text_clutter_metric

app = FastAPI()

model, device = load_model()
ocr_reader = easyocr.Reader(['en'])
obj_yolo = YOLO('yolov8n.pt')

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)  # Renamed for clarity
    h, w, _ = image_np.shape
    img_tensor = torch.tensor(image_np.transpose(2, 0, 1)[None], dtype=torch.float32).to(device)
    centerbias = gaussian_centerbias(h, w, sigma=0.25)
    centerbias = torch.tensor(centerbias, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        log_density = model(img_tensor, centerbias)
        saliency = log_density.squeeze().cpu().numpy()
        saliency_norm = (saliency - saliency.min())/(saliency.max() - saliency.min())

    # text detection
    data = get_text_data(image_np, ocr_reader)
    txt_impact_score = get_txt_impact_score(image_np, data, saliency_norm, max_size=1024)
    gini_score = gini_index_heatmap(saliency_norm)
    entropy = attention_entropy(saliency_norm)
    text_alignment_score = saliency_object_alignment_index(saliency, data.values())
    relative_font_size = relative_font_size_metric(list(data.values()), image_np)
    text_size_variability_score = text_size_variability(list(data.values()))
    text_clutter_metric_score = text_clutter_metric(list(data.values()), w*h)

    # obj detection
    results = obj_yolo.predict(image_np)
    bboxes, labels = get_obj_bboxes_labels(results, obj_yolo)
    object_attention_scores = attention_share_per_object(saliency, bboxes, labels)
    gini_across_objects = gini_index_across_objects(saliency, bboxes)
    object_alignment_score = saliency_object_alignment_index(saliency, bboxes)
    persons_size_ratio_score = persons_size_ratio(bboxes, w*h)

    # Additional color metrics
    avg_color_sat = average_color_saturation(image_np)
    sal_vividness = saliency_weighted_vividness(image_np, saliency_norm)
    color_temp = color_temperature(image_np)

    return {
        "text_impact_score": float(txt_impact_score) if txt_impact_score is not None else 0.0,
        "relative_font_size": float(relative_font_size) if relative_font_size is not None else 0.0,
        "text_font_variability": float(text_size_variability_score) if text_size_variability_score is not None else 0.0,
        "text_clutter_metric": float(text_clutter_metric_score) if text_clutter_metric_score is not None else 0.0,
        "entropy": float(entropy) if entropy is not None else 0.0,
        "gini_score": float(gini_score) if gini_score is not None else 0.0,
        "object_attention_scores": object_attention_scores,
        "gini_index_across_objects": float(gini_across_objects) if gini_across_objects is not None else 0.0,
        "text_alignment_score": float(text_alignment_score) if text_alignment_score is not None else 0.0,
        "object_alignment_score": float(object_alignment_score) if object_alignment_score is not None else 0.0,
        "persons_size_ratio": float(persons_size_ratio_score) if persons_size_ratio_score is not None else 0.0,
        "average_color_saturation": float(avg_color_sat) if avg_color_sat is not None else 0.0,
        "saliency_weighted_vividness": float(sal_vividness) if sal_vividness is not None else 0.0,
        "color_temperature": float(color_temp) if color_temp is not None else 0.0
    }

standard_response = {
    "text_impact_score": 0.0,
    "relative_font_size": 0.0,
    "text_font_variability": 0.0,
    "text_clutter_metric": 0.0,
    "entropy": 0.0,
    "gini_score": 0.0,
    "object_attention_scores": [],
    "gini_index_across_objects": 0.0,
    "text_alignment_score": 0.0,
    "object_alignment_score": 0.0,
    "persons_size_ratio": 0.0,
    "average_color_saturation": 0.0,
    "saliency_weighted_vividness": 0.0,
    "color_temperature": 0.0
    }

@app.post("/update_metrics/")
async def update_metrics(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    h, w, _ = image_np.shape

    # Text metrics (only alignment as per update)
    data = get_text_data(image_np, ocr_reader)
    relative_font_size = relative_font_size_metric(list(data.values()), image_np)
    text_size_variability_score = text_size_variability(list(data.values()))
    text_clutter_metric_score = text_clutter_metric(list(data.values()), w*h)

    # Object metrics (only persons)
    results = obj_yolo.predict(image_np)
    bboxes, labels = get_obj_bboxes_labels(results, obj_yolo)  # Assumes filtering to 'person' in function
    persons_size_ratio_score = persons_size_ratio(bboxes, w*h)

    return {
        "relative_font_size": float(relative_font_size) if relative_font_size is not None else 0.0,
        "text_font_variability": float(text_size_variability_score) if text_size_variability_score is not None else 0.0,
        "text_clutter_metric": float(text_clutter_metric_score) if text_clutter_metric_score is not None else 0.0,
        "persons_size_ratio": float(persons_size_ratio_score) if persons_size_ratio_score is not None else 0.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
