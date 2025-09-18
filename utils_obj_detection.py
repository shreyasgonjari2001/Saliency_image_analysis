import numpy as np

def get_obj_bboxes_labels(results, model):
    """
    Convert YOLO results into inputs for attention_share_per_object.
    """
    # pick first image's results (if single image inference)
    r = results[0]
    # bboxes in xyxy format
    bboxes = r.boxes.xyxy.cpu().numpy().tolist()
    # class labels
    labels = [model.names[int(cls)] for cls in r.boxes.cls.cpu().numpy()]
    # Filter to only 'person' labels
    person_bboxes = []
    person_labels = []
    for bbox, label in zip(bboxes, labels):
        if label == 'person':
            person_bboxes.append(bbox)
            person_labels.append(label)
    return person_bboxes, person_labels

def attention_share_per_object(saliency_map: np.ndarray, bboxes: list, labels: list):
    """
    Calculate attention share per detected object (only for 'person').
    """
    if not bboxes or not labels:
        return None
    total_saliency = saliency_map.sum() + 1e-8
    results = []
    for box, label in zip(bboxes, labels):
        if label != 'person':
            continue  # Skip non-person
        x1, y1, x2, y2 = map(int, box)
        obj_saliency = saliency_map[y1:y2, x1:x2].sum()
        attention_share = obj_saliency / total_saliency
        results.append({
            "label": label,
            "attention_share": float(attention_share)
        })
    return results

def gini_index_across_objects(saliency_map: np.ndarray, bboxes: list):
    """
    Proper Gini index for inequality of attention distribution across objects (only 'person').
    Range: 0 (equal) to 1 (highly unequal).
    """
    if not bboxes:
        return None
    saliencies = []
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        saliencies.append(saliency_map[y1:y2, x1:x2].sum())
    saliencies = np.array(saliencies, dtype=np.float64)
    if saliencies.sum() == 0 or len(saliencies) == 0:
        return 0.0
    # Sort
    sorted_vals = np.sort(saliencies)
    n = len(sorted_vals)
    # Standard Gini formula
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_vals)) / (n * np.sum(sorted_vals))
    return float(abs(gini))

def saliency_object_alignment_index(saliency_map: np.ndarray, bboxes: list):
    if len(bboxes) == 0:
        return 0.0
    h, w = saliency_map.shape
    total_saliency = saliency_map.sum()
    if total_saliency == 0:
        return 1.0
    y_coords, x_coords = np.indices(saliency_map.shape)
    saliency_com_x = np.sum(x_coords * saliency_map) / total_saliency
    saliency_com_y = np.sum(y_coords * saliency_map) / total_saliency
    obj_com_x, obj_com_y = 0.0, 0.0
    total_obj_saliency = 0.0
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        obj_saliency = saliency_map[y1:y2, x1:x2].sum()
        obj_centroid_x = (x1 + x2) / 2
        obj_centroid_y = (y1 + y2) / 2
        obj_com_x += obj_centroid_x * obj_saliency
        obj_com_y += obj_centroid_y * obj_saliency
        total_obj_saliency += obj_saliency
    if total_obj_saliency == 0:
        return 1.0
    obj_com_x /= total_obj_saliency
    obj_com_y /= total_obj_saliency
    dist = np.sqrt((saliency_com_x - obj_com_x)**2 + (saliency_com_y - obj_com_y)**2)
    max_possible_dist = np.sqrt(w**2 + h**2)
    if max_possible_dist == 0:
        return 0.0
    normalized_dist = dist / max_possible_dist
    return float(1 - normalized_dist)

def persons_size_ratio(bboxes: list, image_area) -> float:
    if not bboxes:
        return 0.0
    areas = []
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        obj_area = abs((x2 - x1) * (y2 - y1))
        object_area = obj_area * 0.8 # assume 80% of bbox is person
        areas.append(object_area)

    if not areas or image_area == 0:
        return 0.0
    
    ratios = np.array(areas) / image_area
    return float(np.sum(ratios))
