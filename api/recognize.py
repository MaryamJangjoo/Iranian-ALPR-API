from core.ocr.reconstruct import reconstruct, format_plate, validate_plate


def plate_handler(detections):
    raw = reconstruct(detections)
    formatted = format_plate(raw)

    return {
        "plate_raw": raw,
        "plate": formatted,
        "valid": validate_plate(formatted)
    }