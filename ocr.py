import cv2
import re
import numpy as np

def get_text_data(frame, reader):
    def to_gray(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh

    txt = reader.readtext(to_gray(frame))
    data = {}
    for en in txt:
        coords, text, _ = en
        x_coords = [x[0] for x in coords]
        y_coords = [x[1] for x in coords]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        data[text] = [x1, y1, x2, y2]
    return data

def get_weight(txt: str) -> float:
    """
    Assigns a weight to a string based on whether it contains offer or
    call-to-action (CTA) patterns.
    """
    # Standardize the input text for consistent matching
    clean_txt = txt.strip().lower()

    OFFER_PATTERNS = [
        r"\d+\s?%",
        r"\bbuy\s*1\s*get\s*1\b",
        r"\bfree\b",
        r"\bsave(\s*\d+)?\b",
        r"\bdiscount(s)?\b",
        r"\bsale(s)?\b",
        r"\bdeal(s)?\b",
        r"\blimited(\s*time)?\b",
        r"\btoday(\s*only)?\b",
        r"\s*only\b"
    ]

    CTA_PATTERNS = [
        r"\bshop(\s*now)?\b",
        r"\bbuy(\s*now)?\b",
        r"\border(\s*now)?\b",
        r"\bclick(\s*here)?\b",
        r"\blearn(\s*more)?\b",
        r"\bsign(\s*up)?\b",
        r"\bsubscribe\b",
        r"\bget(\s*started)?\b",
        r"\btry(\s*now)?\b",
        r"\bwatch(\s*now)?\b",
        r"\bdonate(\s*now)?\b",
        r"\bjoin(\s*now)?\b",
    ]

    # Check for offers first (higher weight)
    if any(re.search(pattern, clean_txt) for pattern in OFFER_PATTERNS):
        return 1.5
    # Then check for CTAs
    elif any(re.search(pattern, clean_txt) for pattern in CTA_PATTERNS):
        return 1.2
    # Default weight if no patterns match
    else:
        return 1.0

def get_txt_impact_score(frame, data, saliency_norm, max_size=1024):
    h, w = saliency_norm.shape
    scores = []
    if not data:  # Explicit check for no text
        return 0.0

    for txt, (x1, y1, x2, y2) in data.items():
        #weight = get_weight(txt)
        # convert to ints
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # clip to valid image range
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            frame_small = frame
        if x2 > x1 and y2 > y1:  # valid box
            s = saliency_norm[y1:y2+1, x1:x2+1]
            score = np.mean(s)
            scores.append(score)  # Apply weight here

    if len(scores) == 0:
        return 0.0  # no text detected
    else:
        return float(np.mean(scores))
    
def relative_font_size_metric(text_bboxes: list, frame: int) -> float:
    image_height = frame.shape[0]
    if image_height <= 0:
        return 0.0  # Invalid image height
    
    # Extract heights from bboxes
    heights = []
    for bbox in text_bboxes:
        h = bbox[3] - bbox[1]
        heights.append(abs(h))
    
    if not heights:
        return 0.0  # No text detected: poor score (or return 0.5 if neutral preferred)
    
    # Compute average ratio
    ratios = np.array(heights) / image_height
    return float(np.mean(ratios))

def text_size_variability(text_bboxes: list) -> float:
    if len(text_bboxes) < 2:
        return 0.0  # Not enough text to determine variability
    
    heights = []
    for bbox in text_bboxes:
        h = bbox[3] - bbox[1]
        heights.append(abs(h))
    
    if len(heights) < 2:
        return 0.0  # Not enough valid heights
    
    std_dev = np.std(heights)
    mean_height = np.mean(heights)
    
    if mean_height == 0:
        return 0.0  # Avoid division by zero
    
    variability = std_dev / mean_height
    return float(variability)

def text_clutter_metric(text_bboxes: list, image_area: int) -> float:
    #image_area = frame.shape[0] * frame.shape[1]
    if image_area == 0:
        return 0.0  # Invalid image area
    
    total_text_area = 0
    for bbox in text_bboxes:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        total_text_area += abs(w * h)
    
    if total_text_area == 0:
        return 0.0  # No text detected: poor score (or return 0.5 if neutral preferred)
    
    clutter_ratio = total_text_area / image_area
    return float(clutter_ratio)

