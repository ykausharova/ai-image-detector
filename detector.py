from PIL import Image, ImageOps
from transformers import pipeline

MODELS = {
    "General Detector": "haywoodsloan/ai-image-detector-deploy",
    "SDXL Detector": "Organika/sdxl-detector",
    "Deep Fake Detector": "dima806/ai_vs_real_image_detection",
}

def load_detectors():
    return {name: pipeline("image-classification", model=model_id)
            for name, model_id in MODELS.items()}

# Normalize labels
LABEL_MAP = {
    "human": "real",
    "real": "real",
    "REAL": "real",
    "artificial": "artificial",
    "fake": "artificial",
    "FAKE": "artificial",
}

def analyze_image(image, detectors):
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    image = ImageOps.exif_transpose(image)  # fix camera rotation metadata
    if max(image.size) > 1024: # resize large images
        image.thumbnail((1024, 1024))
    results = {}
    for name, detector in detectors.items():
        output = detector(image)
        scores = {"real": 0.0, "artificial": 0.0}
        for item in output:
            normalized = LABEL_MAP.get(item["label"], item["label"])
            scores[normalized] = round(item["score"], 4)
        results[name] = scores
    return results

WEIGHTS = {
    "General Detector": 0.45,
    "SDXL Detector": 0.35,
    "Deep Fake Detector": 0.20,
}

# Average artificial scores across models for a combined signal
def combined_verdict(results):
    total_weight = sum(WEIGHTS.get(name, 1.0) for name in results)
    avg = sum(WEIGHTS.get(name, 1.0) * scores["artificial"] 
              for name, scores in results.items()) / total_weight
    avg = round(avg, 4)
    label = "ARTIFICIAL" if avg > 0.5 else "REAL"
    return {"verdict": label, "confidence": avg if label == "ARTIFICIAL" else round(1 - avg, 4)}