# Updated utils.py (fix text_alignment_score when no text)

import torch
import numpy as np
import cv2
import deepgaze_pytorch
from PIL import Image
from sklearn.cluster import KMeans

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    model.eval()
    return model, device

def gaussian_centerbias(h, w, sigma=0.25):
    y, x = np.mgrid[0:h, 0:w]
    y = y / h - 0.5
    x = x / w - 0.5
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    g /= g.sum()
    return np.log(g + 1e-8)

def apply_saliency_chroma(image, saliency_norm, alpha=0.7):
    saliency_resized = cv2.resize(saliency_norm, (image.shape[1], image.shape[0]))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saliency_resized = (saliency_resized - saliency_resized.min()) / (saliency_resized.max() - saliency_resized.min() + 1e-8)
    hsv[..., 1] *= (1 - alpha) + alpha * saliency_resized
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out

def show_impact(image, saliency, threshold=0.5):
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    saliency_resized = cv2.resize(saliency_norm, (image.shape[1], image.shape[0]))
    mask = saliency_resized > threshold
    mask = np.repeat(mask[:, :, None], 3, axis=2)
    impacted = np.where(mask, image, 0)
    return impacted

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    return frame

def attention_entropy(saliency_norm: np.ndarray) -> float:
    sal = saliency_norm.flatten()
    p = sal / sal.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)

def gini_index_heatmap(heatmap: np.ndarray) -> float:
    values = heatmap.flatten()
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)

def color_palette_diversity(image: np.ndarray, quantize_levels: int = 32) -> int:
    pil_img = Image.fromarray(image)
    quantized = pil_img.quantize(colors=quantize_levels)
    unique_colors = len(quantized.getcolors(quantize_levels))
    return unique_colors

def dominant_color_ratio(image: np.ndarray, n_clusters: int = 5) -> float:
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_ratio = counts.max() / len(pixels)
    return float(dominant_ratio)

def color_harmony_score(image: np.ndarray, n_colors: int = 5, harmony_threshold: float = 30.0) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)
    hues = kmeans.cluster_centers_[:, 0]  # Hue channel (0-180 in OpenCV)
    harmonious_pairs = 0
    total_pairs = 0
    for i in range(n_colors):
        for j in range(i + 1, n_colors):
            diff = min(abs(hues[i] - hues[j]), 180 - abs(hues[i] - hues[j]))
            if diff <= harmony_threshold or abs(diff - 180) <= harmony_threshold:  # Analogous or complementary
                harmonious_pairs += 1
            total_pairs += 1
    if total_pairs == 0:
        return 0.0
    return float(harmonious_pairs / total_pairs)

def average_color_saturation(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1] / 255.0  # Normalize to [0-1]
    return float(np.mean(saturation))

def saliency_weighted_vividness(image: np.ndarray, saliency_norm: np.ndarray, threshold: float = 0.5) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1] / 255.0
    mask = saliency_norm > threshold
    if np.sum(mask) == 0:
        return 0.0
    weighted_sat = np.mean(saturation[mask])
    return float(weighted_sat)

def color_temperature(image: np.ndarray) -> float:
    b, g, r = cv2.split(image.astype(float))
    warmth = (r - b) / (r + b + 1e-8)  # Avoid division by zero
    return float(np.mean(warmth))
