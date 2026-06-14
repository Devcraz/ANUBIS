import onnxruntime as ort
import numpy as np
from PIL import Image
import os

class GeoCLIP:

    def __init__(self, model_path="models/clip_vision.onnx", embeddings_path="models/text_embeddings.npy", countries_path="models/countries.txt"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.text_embeddings = np.load(embeddings_path)
        with open(countries_path, "r") as f:
            self.countries = [line.strip() for line in f]
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224), Image.BICUBIC)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)
        return img_np.astype(np.float32)

    def encode_image(self, image_path):
        input_tensor = self.preprocess(image_path)
        return self.session.run(None, {self.input_name: input_tensor})[0]

    def predict_country(self, image_path, top_k=5):
        image_embedding = self.encode_image(image_path)
        image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=1, keepdims=True)
        text_embeddings = self.text_embeddings / np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        similarities = np.dot(image_embedding, text_embeddings.T)[0]
        best_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in best_indices:
            results.append({
                "country": self.countries[idx],
                "score": float(similarities[idx])
            })
        return results