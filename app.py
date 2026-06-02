"""
Streamlit demo: type a flower description, get generated images.

    streamlit run app.py

Model is downloaded automatically from HuggingFace Hub on first run.
"""

import os
import torch
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download

from src.model import Generator
from src.text_encoder import TextEncoder, EMBED_DIM
from src.utils import denorm

HF_REPO      = "natsu321/flower-gan-cls"
MODEL_FILE   = "generator_best.pth"
EMBED_OUT    = 128
NOISE_DIM    = 100
LOCAL_PATH   = os.path.join("models", MODEL_FILE)


def get_model_path():
    """Return local path to checkpoint, downloading from HF Hub if needed."""
    if os.path.exists(LOCAL_PATH):
        return LOCAL_PATH
    os.makedirs("models", exist_ok=True)
    return hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE, local_dir="models")


@st.cache_resource
def load_models():
    device = torch.device("cpu")
    ckpt   = get_model_path()
    encoder = TextEncoder(device="cpu")
    G = Generator(channels=3, noise_dim=NOISE_DIM,
                  embed_dim=EMBED_DIM, embed_out_dim=EMBED_OUT)
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()
    return encoder, G


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Flower GAN", page_icon="🌸")
st.title("🌸 Text-to-Flower Generator")
st.caption("GAN-CLS + DistilBERT · Trained on Oxford-102 flowers · 64×64")

prompt = st.text_input(
    "Describe a flower",
    placeholder="e.g. this flower has soft pink petals and a yellow center"
)
n = st.slider("Number of images", 1, 8, 4)

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a description.")
        st.stop()

    with st.spinner("Loading model..."):
        encoder, G = load_models()

    with st.spinner("Generating..."):
        with torch.no_grad():
            emb   = encoder.encode([prompt]).repeat(n, 1)
            noise = torch.randn(n, NOISE_DIM, 1, 1)
            imgs  = denorm(G(noise, emb)).permute(0, 2, 3, 1).numpy()

    cols = st.columns(min(n, 4))
    for i in range(n):
        arr = (imgs[i] * 255).astype("uint8")
        cols[i % len(cols)].image(
            Image.fromarray(arr).resize((192, 192), Image.NEAREST),
            caption=f"#{i+1}"
        )
    st.caption(f'Prompt: "{prompt}"')
