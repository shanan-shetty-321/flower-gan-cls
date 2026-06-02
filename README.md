# Text-to-Image Synthesis — GAN-CLS + DistilBERT (Flowers)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flower-gan-cls-ly29pxv4cappotnbqflhulp.streamlit.app/)

**Live demo:** [flower-gan-cls-ly29pxv4cappotnbqflhulp.streamlit.app](https://flower-gan-cls-ly29pxv4cappotnbqflhulp.streamlit.app/)

Generate 64×64 flower images from natural-language descriptions.
A conditional GAN (DCGAN backbone) trained with the **GAN-CLS** matching-aware
objective, conditioned on frozen **DistilBERT** caption embeddings.

> *Implemented a GAN-CLS architecture with DistilBERT embeddings to generate
> flower images from text. Trained with custom BCE / L1 / L2 losses for
> text–image alignment and training stability.*

---

## How it works

Type a description like *"this flower has soft pink petals and a yellow center"*
and the GAN synthesizes flower images that match it.

| Component | Detail |
|---|---|
| **Text encoder** | Frozen `distilbert-base-uncased`, mean-pooled → 768-d embedding |
| **Generator** | noise(100) ⊕ projected caption → 5× ConvTranspose → 64×64 RGB |
| **Discriminator** | 4× Conv → 4×4×512 features ⊕ tiled caption → real/fake score |
| **Dataset** | [`pranked03/flowers-blip-captions`](https://huggingface.co/datasets/pranked03/flowers-blip-captions) — Oxford-102 flowers + BLIP captions |
| **Trained model** | [`natsu321/flower-gan-cls`](https://huggingface.co/natsu321/flower-gan-cls) on HuggingFace |

## Loss functions (GAN-CLS)

- **Discriminator:** `BCE(real+right_text, 1)` + `BCE(real+wrong_text, 0)` + `BCE(fake, 0)`
  — the *wrong-pair* term forces the model to learn text–image alignment, not just "is this a flower"
- **Generator:** `BCE(fake, 1)` + `10·L1(fake, real)` + `100·L2(feature-matching)`

## Project layout

```
app.py                      Streamlit demo (auto-downloads model from HF Hub)
src/text_encoder.py         DistilBERT -> 768-d embeddings (frozen)
src/dataset.py              GAN-CLS dataset with right/wrong image pairs
src/model.py                Generator + Discriminator
src/utils.py                transforms, weight init, checkpointing
src/train.py                training loop
src/generate.py             CLI inference from a text prompt
configs/train_config.yaml   hyperparameters
kaggle/kaggle_train.ipynb   GPU training notebook (Google Colab / Kaggle)
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or generate directly from the CLI:
```bash
python -m src.generate --prompt "a flower with bright yellow petals" --n 8
```

## Training

Trained on Google Colab (T4 GPU) for 450 epochs, ~94 minutes.
See `kaggle/kaggle_train.ipynb` to reproduce.

## Why not the older reference repos
`paarthneekhara` / `zsdonghao` are TensorFlow 1.x. `aelnouby` uses
legacy char-CNN-RNN embeddings + manual HDF5. This project uses the modern
HuggingFace stack (`datasets` + `transformers`), written clean from scratch.
