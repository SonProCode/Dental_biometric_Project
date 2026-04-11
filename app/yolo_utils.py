"""
Phase 1 – Tooth Segmentation using YOLO.
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

YOLO_MODEL_PATH = Path(__file__).parent.parent / "model" / "yolo_best.pt"

_yolo_model = None


def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def crop_polygon(img: np.ndarray, polygon: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Crop a single tooth from the image using its polygon mask.
    1. Create a binary mask from the polygon.
    2. Crop the bounding rectangle.
    3. Apply mask to preserve tooth shape (black background).
    4. Resize keeping aspect ratio, then pad to `size × size`.
    """
    poly_int = np.array(polygon, dtype=np.int32)

    # --- mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 255)

    # --- bounding rect
    x, y, w, h = cv2.boundingRect(poly_int)
    if w == 0 or h == 0:
        return None

    crop_img = img[y:y + h, x:x + w].copy()
    crop_mask = mask[y:y + h, x:x + w]

    # --- apply mask
    result = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)

    # --- resize keeping aspect ratio
    scale = size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # --- pad to size × size
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


def segment_teeth(image_path: str) -> list[np.ndarray]:
    """
    Run YOLO on `image_path` and return a list of cropped tooth images (128×128 BGR).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    model = get_yolo_model()
    results = model(image_path, conf=0.3, verbose=False)

    teeth = []
    for result in results:
        if result.masks is None:
            continue
        polygons = result.masks.xy          # list of (N, 2) arrays
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            crop = crop_polygon(img, polygon, size=128)
            if crop is not None:
                teeth.append(crop)

    return teeth
