from PIL import Image, ImageOps
from detector import load_detectors, analyze_image, combined_verdict
import streamlit as st

st.set_page_config(page_title="AI Image Detector", page_icon="🔍", layout="centered")

# Cache models so they don't reload on every interaction
@st.cache_resource
def get_detectors():
    return load_detectors()

st.title("🔍 AI Image Detector")
st.write("Upload an image to find out whether it was AI-generated or taken by a human.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if not uploaded_file:
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
except Exception:
    st.error("Could not open the image. Please try a different file.")
    st.stop()

st.image(image, width=600)

with st.spinner("Analyzing..."):
    detectors = get_detectors()
    results = analyze_image(image, detectors)
    verdict = combined_verdict(results)

st.divider()

label = verdict["verdict"]
confidence = verdict["confidence"] * 100

if confidence < 65:
    icon, display_label = "🟡", "UNCERTAIN"
elif label == "ARTIFICIAL":
    icon, display_label = "🔴", "ARTIFICIAL"
else:
    icon, display_label = "🟢", "REAL"

st.markdown(f"## {icon} {display_label}")
st.markdown(f"**Confidence: {confidence:.1f}%**")
st.progress(verdict["confidence"])

st.divider()

st.subheader("Model breakdown")
for model_name, scores in results.items():
    st.markdown(f"**{model_name}**")
    col1, col2 = st.columns(2)
    col1.metric("Real", f"{scores['real']*100:.1f}%")
    col2.metric("Artificial", f"{scores['artificial']*100:.1f}%")

st.divider()

st.warning(
    "⚠️ **Limitations:** Detection accuracy varies by generator. "
    "Results are most reliable for Stable Diffusion images. "
    "No detector is 100% accurate - always apply critical judgment."
)

with st.expander("About this project"):
    st.markdown("""
    This tool runs two independent pre-trained models and combines their verdicts:
    - **General Detector** — trained on a wide variety of AI-generated images
    - **SDXL Detector** — specialized for Stable Diffusion XL outputs

    Running two models in parallel reduces false positives and gives a more reliable
    signal than any single detector alone.

    AI image detection is an active research problem. Modern generators like Midjourney
    produce images that are increasingly hard to detect — this is a known open challenge
    in the field, not a limitation of this tool specifically.

    Built with PyTorch, HuggingFace Transformers, and Streamlit.
    """)