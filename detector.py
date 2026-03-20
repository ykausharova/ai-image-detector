from PIL import Image, ImageOps
from transformers import pipeline

MODELS = {
    "General Detector": "haywoodsloan/ai-image-detector-deploy",
    "SDXL Detector": "Organika/sdxl-detector",
}

def load_detectors():
    return {name: pipeline("image-classification", model=model_id)
            for name, model_id in MODELS.items()}

# Some models use "human" instead of "real": normalize to the same scheme
LABEL_MAP = {
    "human": "real",
    "real": "real",
    "artificial": "artificial",
}

def analyze_image(image, detectors):
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    image = ImageOps.exif_transpose(image)  # fix iPhone/camera rotation metadata
    results = {}
    for name, detector in detectors.items():
        output = detector(image)
        scores = {"real": 0.0, "artificial": 0.0}
        for item in output:
            normalized = LABEL_MAP.get(item["label"], item["label"])
            scores[normalized] = round(item["score"], 4)
        results[name] = scores
    return results

# Average artificial scores across models for a combined signal
def combined_verdict(results):
    scores = [r["artificial"] for r in results.values()]
    avg = round(sum(scores) / len(scores), 4)
    label = "ARTIFICIAL" if avg > 0.5 else "REAL"
    return {"verdict": label, "confidence": avg if label == "ARTIFICIAL" else round(1 - avg, 4)}