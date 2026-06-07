# OCR module for ANUBIS: text extraction, license plate analysis, language detection
import easyocr
import re
import json
import os

class OCR:

    def __init__(self, languages=None, gpu=False):
        if languages is None:
            languages = ['es', 'en', 'fr', 'de', 'it', 'pt']
        self.languages = languages
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.plate_formats = self._load_plate_formats()

    def _load_plate_formats(self):
        json_path = os.path.join(os.path.dirname(__file__), "plate_formats.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # worlwide license plate patterns (fallback)
        return {
            "afghanistan": {"pattern": r"^[A-Z]{1,3}\d{1,5}$"},
            "albania": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "algeria": {"pattern": r"^\d{5,6}\s?[A-Z]{1,3}$"},
            "andorra": {"pattern": r"^[A-Z]{1,2}\d{4}$"},
            "angola": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "argentina": {"pattern": r"^[A-Z]{3}\d{3}$"},
            "armenia": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "australia": {"pattern": r"^[A-Z]{1,3}\d{3}[A-Z]?$"},
            "austria": {"pattern": r"^[A-Z]{1,2}\d{4,5}[A-Z]?$"},
            "azerbaijan": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "bahamas": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "bahrain": {"pattern": r"^\d{5}$"},
            "bangladesh": {"pattern": r"^[A-Z]{2}\d{1,4}$"},
            "barbados": {"pattern": r"^[A-Z]{1,2}\d{1,4}$"},
            "belarus": {"pattern": r"^\d{4}[A-Z]{2}\d{1}$"},
            "belgium": {"pattern": r"^[A-Z]{3}\d{3}$"},
            "belize": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "benin": {"pattern": r"^[A-Z]{1,2}\d{4}[A-Z]{1,2}$"},
            "bhutan": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "bolivia": {"pattern": r"^\d{4}[A-Z]{3}$"},
            "bosnia": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "botswana": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "brazil": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "brunei": {"pattern": r"^[A-Z]{1,2}\d{1,4}$"},
            "bulgaria": {"pattern": r"^[A-Z]{2}\d{4}[A-Z]{2}$"},
            "burkina_faso": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "burundi": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "cambodia": {"pattern": r"^\d{2}[A-Z]{1,2}\d{4}$"},
            "cameroon": {"pattern": r"^[A-Z]{2}\d{4}[A-Z]{1,2}$"},
            "canada": {"pattern": r"^[A-Z]{2}\d{2}\s?[A-Z]{1,2}$"},
            "cape_verde": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "chad": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "chile": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "china": {"pattern": r"^[A-Z]{1}[A-Z0-9]{5}$"},
            "colombia": {"pattern": r"^[A-Z]{3}\d{3}$"},
            "comoros": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "congo": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "costa_rica": {"pattern": r"^\d{5}$"},
            "croatia": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "cuba": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "cyprus": {"pattern": r"^[A-Z]{3}\d{3}$"},
            "czech": {"pattern": r"^\d{2}[A-Z]{2}\d{4}$"},
            "denmark": {"pattern": r"^[A-Z]{2}\d{2}\s?\d{3}$"},
            "djibouti": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "dominican_republic": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "ecuador": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "egypt": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "el_salvador": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "equatorial_guinea": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "eritrea": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "estonia": {"pattern": r"^\d{3}[A-Z]{2}$"},
            "eswatini": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "ethiopia": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "fiji": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "finland": {"pattern": r"^[A-Z]{1,2}\d{3}$"},
            "france": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "gabon": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "gambia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "georgia": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "germany": {"pattern": r"^[A-Z]{1,3}\s?[A-Z]{1,2}\s?\d{1,4}$"},
            "ghana": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "greece": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "grenada": {"pattern": r"^[A-Z]{1,2}\d{1,4}$"},
            "guatemala": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "guinea": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "guinea_bissau": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "guyana": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "haiti": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "honduras": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "hungary": {"pattern": r"^[A-Z]{3}\d{3}$"},
            "iceland": {"pattern": r"^[A-Z]{2}\d{3}$"},
            "india": {"pattern": r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$"},
            "indonesia": {"pattern": r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,2}$"},
            "iran": {"pattern": r"^\d{2}[A-Z]\d{3}\d{2}$"},
            "iraq": {"pattern": r"^\d{1,6}$"},
            "ireland": {"pattern": r"^\d{2}[A-Z]{2}\d{4}$"},
            "israel": {"pattern": r"^\d{7,8}$"},
            "italy": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "jamaica": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "japan": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "jordan": {"pattern": r"^\d{6}$"},
            "kazakhstan": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "kenya": {"pattern": r"^[A-Z]{3}\d{3}[A-Z]$"},
            "kiribati": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "kosovo": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "kuwait": {"pattern": r"^\d{5}$"},
            "kyrgyzstan": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "laos": {"pattern": r"^\d{2}[A-Z]{1,2}\d{4}$"},
            "latvia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "lebanon": {"pattern": r"^\d{5}$"},
            "lesotho": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "liberia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "libya": {"pattern": r"^\d{5}$"},
            "liechtenstein": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "lithuania": {"pattern": r"^[A-Z]{3}\d{3}$"},
            "luxembourg": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "madagascar": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "malawi": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "malaysia": {"pattern": r"^[A-Z]{1,2}\d{1,4}$"},
            "maldives": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "mali": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "malta": {"pattern": r"^[A-Z]{3}\d{3}$"},
            "marshall_islands": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "mauritania": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "mauritius": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "mexico": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "micronesia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "moldova": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "monaco": {"pattern": r"^[A-Z]{1,2}\d{3}$"},
            "mongolia": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "montenegro": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "morocco": {"pattern": r"^\d{5,6}\s?[A-Z]{1,2}$"},
            "mozambique": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "myanmar": {"pattern": r"^\d{1,2}[A-Z]{1,2}\d{4}$"},
            "namibia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "nauru": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "nepal": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "netherlands": {"pattern": r"^[A-Z]{2}\d{2}[A-Z]{2}$"},
            "new_zealand": {"pattern": r"^[A-Z]{1,3}\d{1,4}$"},
            "nicaragua": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "niger": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "nigeria": {"pattern": r"^[A-Z]{3}\d{3}[A-Z]{2}$"},
            "north_korea": {"pattern": r"^\d{2}[A-Z]{1,2}\d{4}$"},
            "north_macedonia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "norway": {"pattern": r"^[A-Z]{2}\d{5}$"},
            "oman": {"pattern": r"^\d{5}$"},
            "pakistan": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "palau": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "panama": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "papua_new_guinea": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "paraguay": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "peru": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "philippines": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "poland": {"pattern": r"^[A-Z]{2}\d{5}$"},
            "portugal": {"pattern": r"^\d{2}[A-Z]{2}\d{2}$"},
            "qatar": {"pattern": r"^\d{6}$"},
            "romania": {"pattern": r"^[A-Z]{2}\d{2}[A-Z]{3}$"},
            "russia": {"pattern": r"^[A-Z]{1}\d{3}[A-Z]{2}\d{2,3}$"},
            "rwanda": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "saint_kitts": {"pattern": r"^[A-Z]{1,2}\d{1,4}$"},
            "saint_lucia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "samoa": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "san_marino": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "saudi_arabia": {"pattern": r"^\d{4}[A-Z]{3}$"},
            "senegal": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "serbia": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "seychelles": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "sierra_leone": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "singapore": {"pattern": r"^[A-Z]{1,2}\d{1,4}[A-Z]?$"},
            "slovakia": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "slovenia": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]{2}$"},
            "solomon_islands": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "somalia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "south_africa": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "south_korea": {"pattern": r"^\d{2}[A-Z]{1,2}\d{4}$"},
            "south_sudan": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "spain": {"pattern": r"^\d{4}\s?[A-Z]{3}$"},
            "sri_lanka": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "sudan": {"pattern": r"^\d{5}$"},
            "suriname": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "sweden": {"pattern": r"^[A-Z]{3}\d{3}$"},
            "switzerland": {"pattern": r"^[A-Z]{2}\d{5}$"},
            "syria": {"pattern": r"^\d{6}$"},
            "taiwan": {"pattern": r"^\d{2}[A-Z]{1,2}\d{4}$"},
            "tajikistan": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "tanzania": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "thailand": {"pattern": r"^\d{2}[A-Z]{2}\d{4}$"},
            "timor_leste": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "togo": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "tonga": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "trinidad": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "tunisia": {"pattern": r"^\d{4}[A-Z]{2}$"},
            "turkey": {"pattern": r"^\d{2}[A-Z]{1,3}\d{4}$"},
            "turkmenistan": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "tuvalu": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "uganda": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "ukraine": {"pattern": r"^[A-Z]{2}\d{4}[A-Z]{2}$"},
            "united_arab_emirates": {"pattern": r"^\d{5}$"},
            "united_kingdom": {"pattern": r"^[A-Z]{2}\d{2}\s?[A-Z]{3}$"},
            "united_states": {"pattern": r"^[A-Z]{2}\d{3}[A-Z]?$"},
            "uruguay": {"pattern": r"^[A-Z]{3}\d{4}$"},
            "uzbekistan": {"pattern": r"^\d{2}[A-Z]{2}\d{3}$"},
            "vanuatu": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "vatican": {"pattern": r"^SCV\d{1,5}$"},
            "venezuela": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "vietnam": {"pattern": r"^\d{2}[A-Z]{1,2}\d{5}$"},
            "yemen": {"pattern": r"^\d{5}$"},
            "zambia": {"pattern": r"^[A-Z]{2}\d{4}$"},
            "zimbabwe": {"pattern": r"^[A-Z]{2}\d{4}$"},
        }

    def extract_text(self, image):
        is isinstance(image, str):
            return self.reader.readtext(image)
        return self.reader.readtext(image)

    # identify country and region from a license plate text
    def classify_plate(self, text):
        text = text.strip().upper().replace(" ", "").replace("-", "")
        for country, info in self.plate_formats.items():
            if re.match(info["pattern"], text):
                region = None
                regions = info.get("regions", {})
                for prefix, name in regions.items():
                    if text.startswith(prefix):
                        region = name
                        break  
                return {"country": country, "region": region}
        return None

    # guess language from common words found in signs and street names
    def detect_language(self, text):
        text = text.lower()
        if any(w in text for w in ["el", "la", "los", "las", "de", "en", "calle"]):
            return "es"
        if any(w in text for w in ["the", "street", "road", "avenue"]):
            return "en"
        if any(w in text for w in ["le", "la", "rue", "avenue", "du"]):
            return "fr"
        if any(w in text for w in ["der", "die", "das", "straße"]):
            return "de"
        if any(w in text for w in ["il", "la", "via", "piazza"]):
            return "it"
        if any(w in text for w in ["o", "a", "rua", "avenida"]):
            return "pt"
        return "unknown"

    def process_image(self, image_path):
        results = self.extract_text(image_path)
        texts = []
        plates = []
        languages = []
        for (bbox, text, confidence) in results:
            if confidence < 0.3:
                continue
            texts.append({"text": text, "confidence": confidence, "bbox": bbox})
            plate_info = self.classify_plate(text)
            if plate_info.
                plates.append({"text": text, "country": plate_info["country"], "region": plate_info.get("region")})
            lang = self.detect_language(text)
            languages.append(lang)
        detected_countries = list(set(p["country"] for p in plates))
        detected_langs = list(set(languages))
        return {
            "texts": texts, 
            "plates": plates,
            "languages": detected_langs,
            "country_hints": detected_countries,
            "total_text_items": len(texts)
        }