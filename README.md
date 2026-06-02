# Text-to-Image Synthesis — GAN-CLS + DistilBERT (Flowers)

Generate photorealistic 64×64 flower images from natural-language descriptions.
A conditional GAN (DCGAN backbone) is trained with the **GAN-CLS** matching-aware
objective, conditioned on **DistilBERT** caption embeddings.

> *Implemented a GAN-CLS architecture with DistilBERT embeddings to generate
> flower images from text. Trained with custom BCE / L1 / L2 losses for
> text–image alignment and training stability.*

## Approach

- **Dataset:** [`pranked03/flowers-blip-captions`](https://huggingface.co/datasets/pranked03/flowers-blip-captions) — Oxford-102 flowers + BLIP captions (HuggingFace `datasets`, one-line load).
- **Text encoder:** frozen `distilbert-base-uncased`, mean-pooled → 768-d vector (`src/text_encoder.py`).
- **Generator:** noise(100) ⊕ projected-caption → ConvTranspose stack → 64×64 RGB (`src/model.py`).
- **Discriminator:** image → 4×4×512 features ⊕ tiled caption → real/fake score.
- **GAN-CLS losses** (`src/train.py`):
  - **D** = `BCE(real,1)` + `BCE(wrong-pair,0)` + `BCE(fake,0)` — the *wrong pair* (real image + mismatched caption) is what teaches text–image alignment.
  - **G** = `BCE(fake,1)` + `50·L1(fake,real)` + `100·L2(feature-matching)`.

## Layout

```
configs/train_config.yaml   hyperparameters
src/text_encoder.py         DistilBERT -> 768-d embeddings (frozen)
src/dataset.py              GAN-CLS dataset (right/wrong pairs)
src/model.py                Generator + Discriminator
src/utils.py                transforms, init, checkpointing, sample grids
src/train.py                training loop
src/generate.py             inference from a text prompt
app.py                      Streamlit demo
smoke_test.py               local CPU pipeline sanity check
kaggle/kaggle_train.ipynb   GPU training notebook for Kaggle
flower/                     local CPU virtualenv (Python 3.12)
```

## Workflow (this machine has no GPU)

**1. Local — develop & sanity-check (CPU):**
```powershell
flower\Scripts\python.exe smoke_test.py        # ~30s: proves the pipeline works
```

**2. Kaggle — train (free T4/P100 GPU):** see `kaggle/kaggle_train.ipynb`.
Zip the project (exclude `flower/`), upload as a Kaggle Dataset, run the notebook,
download `models/generator_latest.pth`.

**3. Local — demo (CPU):** drop the `.pth` into `models/`, then:
```powershell
flower\Scripts\python.exe -m src.generate --prompt "a flower with pink petals" --n 8
flower\Scripts\streamlit run app.py
```

## Why not the older reference repos
`paarthneekhara` / `zsdonghao` are TensorFlow 1.x (dead in 2026). `aelnouby` uses
legacy char-CNN-RNN embeddings + manual HDF5. This project follows the modern
`nhanth301` approach (HF `datasets` + `transformers`), rewritten clean.
