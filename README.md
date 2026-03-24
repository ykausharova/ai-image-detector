---
title: AI Image Detector
emoji: 🔍
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.44.0
app_file: app.py
pinned: false
---

# 🔍 AI Image Detector

A web app that detects whether an image is AI-generated or taken by a human.

Try it live -> [huggingface.co/spaces/ykausharova/ai-image-detector](https://huggingface.co/spaces/ykausharova/ai-image-detector)

## How it works

Three independent classifiers analyze the image and their confidence scores are
combined using a weighted average. Giving more influence to broader, more general
detectors reduces false positives and makes the final verdict more reliable than
any single model alone.

**Models used:**
- `haywoodsloan/ai-image-detector-deploy` — trained on a wide variety of AI-generated images
- `Organika/sdxl-detector` — specialized for Stable Diffusion XL outputs
- `dima806/ai_vs_real_image_detection` — Vision Transformer trained on a large real/fake dataset

## Stack

Python, PyTorch, HuggingFace Transformers, Streamlit

## Run locally
```bash
git clone https://github.com/ykausharova/ai-image-detector
cd ai-image-detector
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

Detection accuracy varies depending on the generator used. Models perform best
on Stable Diffusion images and may struggle with newer generators like Midjourney.
No detector is 100% accurate so results should always be interpreted with judgment.

AI image detection is an active research problem - as generators improve, detectors
need to be retrained to keep up.