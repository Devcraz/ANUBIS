# OCR module for ANUBIS: basic Western language text extraction, plate classification
import easyocr
import re
import json
import os

# Only Western languages that EasyOCR loads together without conflicts
SAFE_LANGS = ['en', 'es', 'fr', 'de', 'it', 'pt']

class OCRReader:
    def __init__(self, languages=None, gpu=False):
        if languages is None:
            languages = SAFE_LANGS
        self.languages = languages
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
        }

    def extract_text(self, image_path):
        return self.reader.readtext(image_path)

    def classify_plate(self, text):
        text = text.strip().upper().replace(" ", "").replace("-", "")
        for country, info in self.plate_formats.items():
            if re.match(info["pattern"], text):
                return {"country": country, "region": None}
        return None

    def process_image(self, image_path):
        results = self.extract_text(image_path)
        texts = []
        plates = []
        langs = []
        for (bbox, text, conf) in results:
            if conf < 0.3:
                continue
            lang = self.reader.detect_language(text) if hasattr(self.reader, 'detect_language') else "unknown"
            texts.append({"text": text, "confidence": conf, "language": lang})
            plate = self.classify_plate(text)
            if plate:
                plates.append(plate)
            langs.append(lang)
        return {
            "texts": texts,
            "plates": plates,
            "languages": list(set(langs))
        }