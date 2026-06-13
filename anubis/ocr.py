# OCR module for ANUBIS: license plate and language detection
import easyocr
import re
import json
import os

class OCRReader:
    def __init__(self, languages=None, gpu=False):
        if languages is None:
            languages = ['es', 'en', 'fr', 'de', 'it', 'pt']
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.plate_formats = self._load_plate_formats()

    def _load_plate_formats(self):
        json_path = os.path.join(os.path.dirname(__file__), "plate_formats.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "spain": {"pattern": r"^\d{4}\s?[A-Z]{3}$"},
            "germany": {"pattern": r"^[A-Z]{1,3}\s?[A-Z]{1,2}\s?\d{1,4}$"},
            "france": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "italy": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "united_kingdom": {"pattern": r"^[A-Z]{2}\d{2}\s?[A-Z]{3}$"},
            "united_states": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]?$"},
            "portugal": {"pattern": r"^\d{2}[A-Z]{2}\d{2}$"},
            "japan": {"pattern": r"^[A-Z]{2}\d{4}$"},
        }

    def extract_text(self, image_path):
        return self.reader.readtext(image_path)

    # classify plate text into country and optional region
    def classify_plate(self, text):
        text = text.strip().upper().replace(" ", "").replace("-", "")
        for country, info in self.plate_formats.items():
            if re.match(info["pattern"], text):
                return {"country": country, "region": None}
        return None

    # guess language from common words
    def detect_language(self, text):
        text = text.lower()
        if any(w in text for w in ["el", "la", "los", "calle"]):
            return "es"
        if any(w in text for w in ["the", "street", "road"]):
            return "en"
        if any(w in text for w in ["le", "la", "rue"]):
            return "fr"
        if any(w in text for w in ["der", "die", "straße"]):
            return "de"
        if any(w in text for w in ["il", "la", "via"]):
            return "it"
        if any(w in text for w in ["o", "a", "rua"]):
            return "pt"
        return "unknown"

    def process_image(self, image_path):
        results = self.extract_text(image_path)
        texts = []
        plates = []
        langs = []
        for (bbox, text, conf) in results:
            if conf < 0.3:
                continue
            texts.append({"text": text, "confidence": conf})
            plate = self.classify_plate(text)
            if plate:
                plates.append(plate)
            langs.append(self.detect_language(text))
        return {
            "texts": texts,
            "plates": plates,
            "languages": list(set(langs))
        }