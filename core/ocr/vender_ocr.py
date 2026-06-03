from core.interfaces.ocr import BaseOCR


class VendorOCR(BaseOCR):

    def __init__(self):
        # Ø§ÛŒÙ†Ø¬Ø§ SDK Ø´Ø±Ú©Øª ÛŒØ§ API client Ù…ÛŒØ§Ø¯
        pass

    def recognize(self, image):

        # ÙØ±Ø¶: API ÛŒØ§ DLL ÛŒØ§ REST call
        return [{
            "text": "12Ø§Ù„Ù34567",
            "confidence": 0.98,
            "bbox": [0, 0, 100, 50]
        }]
