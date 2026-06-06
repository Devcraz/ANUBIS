import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Models
MODELS_DIR = os.path.join(BASE_DIR, "models")
VEGETATION_MODEL = os.path.join(MODELS_DIR, "vegetation_mobilenetv2.onnx")
PLACES_MODEL = os.path.join(MODELS_DIR, "places365_resnet18.onnx")
SIGNS_MODEL = os.path.join(MODELS_DIR, "yolov5s_signs.onnx")

#Cache for OSM queries and geocoding
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
CACHE_EXPIRY = 60 * 60 * 24 * 7

#OCR settings
OCR_LANGUAGES = ['es', 'en', 'fr', 'de', 'it', 'pt']
OCR_MIN_CONFIDENCE = 0.5

#Solar analysis
SOLAR_GRID_STEP = 0.5

#Map queries
OSM_USER_AGENT = "ANUBIS-geoint"
NOMINATIM_USER_AGENT = "ANUBIS-geoint"

#Fusion
FUSION_PARTICLE_COUNT = 10000
FUSION_TOP_K = 5 # how many final locations to output