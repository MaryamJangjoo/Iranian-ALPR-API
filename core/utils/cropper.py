import cv2

def crop_plate(frame, bbox, resize_width=320):

    if bbox is None:
        return None

    if isinstance(bbox, dict):
        bbox = bbox.get("bbox", None)

    if bbox is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)

    h_img, w_img = frame.shape[:2]

    # ?? FIX 1: expand bbox (VERY IMPORTANT)
    w = x2 - x1
    h = y2 - y1

    pad_x = int(w * 0.25)   # 25% expansion
    pad_y = int(h * 0.25)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # ?? FIX 2: normalize aspect ratio (very important for OCR)
    h, w = crop.shape[:2]

    if w == 0:
        return None

    scale = resize_width / w
    new_h = int(h * scale)

    crop = cv2.resize(crop, (resize_width, new_h))

    return crop