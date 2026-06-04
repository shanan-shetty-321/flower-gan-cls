# Project Conversation Context — GAN-CLS + DistilBERT Flower Generator

This file captures the full context of the conversation between Shanan Shetty and Claude (Opus 4.8 + Sonnet 4.6) that built this project from scratch. Send this file to any AI agent to resume with full context.

---

## Who is the user

- **Name:** Shanan Shetty
- **Email:** shanan@propheus.com
- **GitHub:** https://github.com/shanan-shetty-321
- **HuggingFace:** https://huggingface.co/natsu321
- **Goal:** Portfolio/learning project — wants to understand the code, not just run it
- **Communication style:** Casual. Prefers concrete explanations of the *why*, not just the *how*

---

## Machine specs

| Property | Value |
|---|---|
| OS | Windows 11 Pro |
| CPU | Intel Core i5-10310U @ 1.70GHz |
| RAM | 16 GB |
| GPU | Intel UHD Graphics only — **NO NVIDIA / NO CUDA** |
| Python | 3.12.9 (venv: `flower/`) + system 3.13.6 |
| Shell | PowerShell |
| C: drive free | ~0.7 GB (critically low) |
| D: drive free | ~31 GB |

**Critical:** `HF_HOME` is permanently set to `D:\hf_cache` (user env var) so all HuggingFace downloads go to D: not C:.

---

## Project location

```
C:\Users\LENOVO\Downloads\flower_project\
```

Venv: `flower\Scripts\python.exe` (never needs activation, call directly)

---

## What was built

**Text-to-Image Synthesis** — type a flower description, get a 64×64 flower image.

### Architecture

- **Text encoder:** Frozen `distilbert-base-uncased` (HuggingFace transformers), mean-pooled → **768-d embedding**. Frozen = never backprop into BERT, just use its features.
- **Generator (DCGAN):** caption(768) → Linear → 128-d projection ⊕ noise(100) → 5× ConvTranspose2d layers (512→256→128→64→3) → Tanh → **64×64 RGB image** in [-1,1]
- **Discriminator:** 64×64 image → 4× Conv2d (64→128→256→512) → 4×4×512 feature map → caption(128) tiled over 4×4 grid → concatenate → Conv2d → Sigmoid → scalar. Returns `(prob, features)`.
- **GAN-CLS trick (Reed et al. 2016):** Discriminator trains on 3 pairs:
  - real image + matching caption → label 1 (real & aligned)
  - real image + **wrong caption** → label 0 (mismatched) ← this forces text-image alignment
  - fake image + matching caption → label 0 (fake)
- **Losses:**
  - D = `BCE(real, 0.9)` + `BCE(wrong_pair, 0)` + `BCE(fake, 0)` (0.9 = label smoothing)
  - G = `BCE(fake, 1)` + `10·L1(fake, real)` + `100·L2(feature_matching)`
  - Feature matching L2: push mean discriminator features of fake images toward real images → stabilizes training

### Dataset

`pranked03/flowers-blip-captions` on HuggingFace — Oxford-102 flowers with BLIP-generated captions. 6552 samples. Loads in one line via `datasets` library, no manual download needed.

### Stack (2026-compatible, no TensorFlow)

```
torch 2.12.0+cpu (local) / torch latest (Colab/Streamlit Cloud)
torchvision
transformers 5.9.0
datasets 4.8.5
streamlit 1.58.0
huggingface_hub
pyyaml, pillow, numpy, tqdm, matplotlib, accelerate
```

---

## Key decisions made

| Decision | Choice | Reason |
|---|---|---|
| Training venue | **Google Colab** (T4 GPU) | No local GPU; Colab free T4 sufficient |
| Build style | **Clean guided rewrite** | Portfolio/learning — not a direct clone |
| Reference repo | `nhanth301/Text2Image-GANCLS-BERT` | Only modern (non-TF) implementation of the 8 reviewed |
| Text encoder | DistilBERT frozen | Stable, 768-d, mean-pooled, no training instability |
| Resolution | 64×64 | Standard for this architecture; higher needs StackGAN |
| Epochs | 450 | ~94 min on T4; good quality without diminishing returns |
| L1_COEFF | 10 (not 50) | 50 caused blurriness + mode collapse; 10 balances sharpness |
| REAL_LABEL | 0.9 | Label smoothing — prevents discriminator from dominating |
| Model hosting | HuggingFace Hub (`natsu321/flower-gan-cls`) | Standard for ML demos |
| Deployment | Streamlit Community Cloud | Free, deploys from GitHub, ML-friendly |

---

## Why the other 7 repos were rejected

| Repo | Problem |
|---|---|
| `paarthneekhara/text-to-image` | TensorFlow 1.x — dead in 2026 |
| `zsdonghao/text-to-image` | TensorFlow + TensorLayer — dead |
| `aelnouby/Text-to-Image-Synthesis` | char-CNN-RNN embeddings (not BERT) + manual HDF5 |
| `ypxie/HDGan` | StackGAN-style, much heavier, different approach |
| `ali-vilab/Cones-V2` | Diffusion-based (not GAN-CLS), very heavy |
| `ShanHaoYu/Text2Image` | Older/partial |
| `Ereebay/text2img` | Older/partial |
| `Fahmaan114/hello` | Unrelated |

---

## File structure

```
flower_project/
├── app.py                        # Streamlit demo — auto-downloads model from HF Hub
├── smoke_test.py                 # Local CPU pipeline sanity check (2 training steps)
├── requirements.txt              # No version pins on torch (Streamlit Cloud Python 3.14)
├── README.md                     # Portfolio README with live demo badge
├── CLAUDE.md                     # AI handoff doc (architecture + decisions)
├── CONVERSATION_CONTEXT.md       # This file
├── .gitignore                    # Excludes flower/, models/, outputs/
├── configs/
│   └── train_config.yaml         # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── text_encoder.py           # DistilBERT TextEncoder class (frozen, mean-pooled)
│   ├── dataset.py                # Text2ImageDataset (right/wrong pairs, RAM preload)
│   ├── model.py                  # Generator + Discriminator (text_proj layer name)
│   ├── utils.py                  # transform, weights_init, denorm, save_sample_grid
│   ├── train.py                  # Full training loop with best-model checkpoint
│   └── generate.py               # CLI inference: --prompt "..." --n 8
├── kaggle/
│   └── kaggle_train.ipynb        # Self-contained Colab/Kaggle training notebook
├── models/
│   └── generator_best.pth        # Trained weights (gitignored, hosted on HF Hub)
└── flower/                       # Python 3.12.9 venv (gitignored)
```

---

## Critical naming note

The Generator and Discriminator both have a text projection layer named **`text_proj`** (not `text_embedding`). This must match between the Colab training code and the local `src/model.py` — otherwise `load_state_dict` will crash with a key-mismatch error when loading the trained checkpoint locally.

---

## Trained model

- **HuggingFace:** https://huggingface.co/natsu321/flower-gan-cls
- **File:** `generator_best.pth` (18 MB)
- **Trained:** 450 epochs on Google Colab T4 GPU, ~94 minutes
- **Config:** batch=64, lr=2e-4, L1=10, L2=100, label_smoothing=0.9, Adam β(0.5, 0.999)

---

## Live deployment

- **Streamlit app:** https://flower-gan-cls-ly29pxv4cappotnbqflhulp.streamlit.app/
- **GitHub repo:** https://github.com/shanan-shetty-321/flower-gan-cls
- Model auto-downloads from HF Hub on first boot (~2 min), cached after that

---

## Google Colab training cells (final version)

### Cell 1 — Install
```python
!pip install -q transformers datasets pyyaml
```

### Cell 2 — Config
```python
import os, time, torch, torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATASET      = "pranked03/flowers-blip-captions"
TEXT_MODEL   = "distilbert-base-uncased"
EMBED_DIM    = 768
EMBED_OUT    = 128
NOISE_DIM    = 100
BATCH_SIZE   = 64
LR           = 0.0002
EPOCHS       = 450
L1_COEFF     = 10
L2_COEFF     = 100
REAL_LABEL   = 0.9
SAMPLE_EVERY = 10
SAVE_EVERY   = 50
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
```

### Cell 3 — Transform
```python
IMAGE_SIZE = 64
transform = T.Compose([
    T.Resize(IMAGE_SIZE), T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
])
def denorm(t): return (t.clamp(-1, 1) + 1) / 2
```

### Cell 4 — Text Encoder
```python
class TextEncoder:
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL)
        self.model = DistilBertModel.from_pretrained(TEXT_MODEL).to(DEVICE).eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts, max_length=64):
        if isinstance(texts, str): texts = [texts]
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt").to(DEVICE)
        hidden = self.model(**enc).last_hidden_state
        mask   = enc["attention_mask"].unsqueeze(-1).float()
        return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu()
```

### Cell 5 — Dataset (images preloaded into RAM)
```python
class Text2ImageDataset(Dataset):
    def __init__(self, hf_split, embeddings):
        print("Preloading all images into RAM...")
        self.embeddings = embeddings
        self.labels     = list(hf_split["label"])
        self.n          = len(self.labels)
        self.images = []
        for i, row in enumerate(hf_split):
            img = row["image"]
            if img.mode != "RGB": img = img.convert("RGB")
            self.images.append(transform(img))
            if i % 1000 == 0: print(f"  {i}/{self.n}")
        self.images = torch.stack(self.images)
        print(f"Done: {tuple(self.images.shape)}")

    def __len__(self): return self.n

    def _wrong_idx(self, label):
        j = np.random.randint(self.n)
        while self.labels[j] == label: j = np.random.randint(self.n)
        return j

    def __getitem__(self, idx):
        return {
            "right_images": self.images[idx],
            "right_embed":  self.embeddings[idx],
            "wrong_images": self.images[self._wrong_idx(self.labels[idx])],
            "txt": "",
        }
```

### Cell 6 — Generator + Discriminator
```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_OUT),
            nn.BatchNorm1d(EMBED_OUT),
            nn.LeakyReLU(0.2, inplace=True),
        )
        def block(ic, oc, k=4, s=2, p=1, last=False):
            layers = [nn.ConvTranspose2d(ic, oc, k, s, p, bias=False)]
            layers += [nn.Tanh()] if last else [nn.BatchNorm2d(oc), nn.ReLU(True)]
            return layers
        self.model = nn.Sequential(
            *block(NOISE_DIM+EMBED_OUT, 512, s=1, p=0),
            *block(512, 256), *block(256, 128), *block(128, 64),
            *block(64, 3, last=True),
        )
    def forward(self, noise, text):
        t = self.text_proj(text).view(-1, EMBED_OUT, 1, 1)
        return self.model(torch.cat([t, noise], 1))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ic, oc, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, 2, 1, bias=False)]
            if norm: layers += [nn.BatchNorm2d(oc)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers
        self.features  = nn.Sequential(
            *block(3, 64, norm=False), *block(64, 128),
            *block(128, 256), *block(256, 512),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_OUT),
            nn.BatchNorm1d(EMBED_OUT),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512+EMBED_OUT, 1, 4, 1, 0, bias=False), nn.Sigmoid(),
        )
    def forward(self, img, text):
        f = self.features(img)
        t = self.text_proj(text).view(-1, EMBED_OUT, 1, 1).repeat(1, 1, 4, 4)
        return self.classifier(torch.cat([f, t], 1)).view(-1), f

def weights_init(m):
    if "Conv" in m.__class__.__name__: nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in m.__class__.__name__:
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.constant_(m.bias, 0)
```

### Cell 7 — Load data + embeddings
```python
ds      = load_dataset(DATASET, split="train")
encoder = TextEncoder()

print("Encoding captions...")
all_embeddings = []
for i in range(0, len(ds), 64):
    all_embeddings.append(encoder.encode(ds["text"][i:i+64]))
    if i % 640 == 0: print(f"  {i}/{len(ds)}")
embeddings = torch.cat(all_embeddings, dim=0)
print(f"Embeddings: {tuple(embeddings.shape)}")

dataset = Text2ImageDataset(ds, embeddings)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)
print(f"{len(dataset)} samples, {len(loader)} batches/epoch")
```

### Cell 8 — Training loop
```python
G = Generator().to(DEVICE);     G.apply(weights_init)
D = Discriminator().to(DEVICE); D.apply(weights_init)
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
bce, l1, l2 = nn.BCELoss(), nn.L1Loss(), nn.MSELoss()

os.makedirs("samples", exist_ok=True)
os.makedirs("models",  exist_ok=True)
fixed_noise = torch.randn(16, NOISE_DIM, 1, 1, device=DEVICE)
fixed_embed = embeddings[:16].to(DEVICE)
best_g_loss = float("inf")

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    d_sum = g_sum = 0
    for batch in loader:
        real  = batch["right_images"].to(DEVICE)
        wrong = batch["wrong_images"].to(DEVICE)
        emb   = batch["right_embed"].to(DEVICE)
        bs    = real.size(0)
        ones  = torch.full((bs,), REAL_LABEL, device=DEVICE)
        zeros = torch.zeros(bs, device=DEVICE)

        opt_D.zero_grad()
        fake        = G(torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE), emb)
        real_p,  _  = D(real, emb)
        wrong_p, _  = D(wrong, emb)
        fake_p,  _  = D(fake.detach(), emb)
        d_loss = bce(real_p, ones) + bce(wrong_p, zeros) + bce(fake_p, zeros)
        d_loss.backward(); opt_D.step()

        opt_G.zero_grad()
        fake        = G(torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE), emb)
        fake_p, f_f = D(fake, emb)
        _,      f_r = D(real, emb)
        g_loss = (bce(fake_p, torch.ones(bs, device=DEVICE))
                  + L1_COEFF * l1(fake, real)
                  + L2_COEFF * l2(f_f.mean(0), f_r.mean(0).detach()))
        g_loss.backward(); opt_G.step()
        d_sum += d_loss.item(); g_sum += g_loss.item()

    n = len(loader); avg_d = d_sum/n; avg_g = g_sum/n
    is_best = avg_g < best_g_loss
    if is_best:
        best_g_loss = avg_g
        torch.save(G.state_dict(), "models/generator_best.pth")
        torch.save(D.state_dict(), "models/discriminator_best.pth")
    print(f"[{epoch:03d}/{EPOCHS}] D={avg_d:.3f}  G={avg_g:.3f}  "
          f"({'BEST ' if is_best else '      '}){time.time()-t0:.1f}s")
    if epoch % SAMPLE_EVERY == 0 or epoch == 1:
        G.eval()
        with torch.no_grad(): imgs = G(fixed_noise, fixed_embed)
        vutils.save_image(denorm(imgs), f"samples/epoch_{epoch:04d}.png", nrow=4)
        G.train()
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        torch.save(G.state_dict(), f"models/generator_epoch_{epoch:04d}.pth")
        torch.save(G.state_dict(), "models/generator_latest.pth")
        torch.save(D.state_dict(), "models/discriminator_latest.pth")

print(f"Done! Best G loss: {best_g_loss:.3f}")
```

### Cell 9 — Inference
```python
G.eval()
prompt = "this flower has bright yellow petals with a dark center"
emb    = encoder.encode([prompt]).repeat(8, 1).to(DEVICE)
noise  = torch.randn(8, NOISE_DIM, 1, 1, device=DEVICE)
with torch.no_grad():
    imgs = denorm(G(noise, emb)).permute(0, 2, 3, 1).cpu().numpy()
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow((imgs[i]*255).astype("uint8")); ax.axis("off")
plt.suptitle(prompt); plt.tight_layout(); plt.show()
```

---

## Issues encountered and fixes

| Issue | Fix |
|---|---|
| First epoch took 400s (not ~12s) | DataLoader was decoding images from parquet on every `__getitem__`. Fixed by preloading all 6552 images into a RAM tensor in `__init__`. |
| C: drive full (0.7 GB free) | Set `HF_HOME=D:\hf_cache` permanently as user env var |
| `generator_latest.pth` not found locally | Changed default checkpoint to `generator_best.pth` in `generate.py` and `app.py` |
| `text_embedding` vs `text_proj` naming mismatch | Fixed `src/model.py` to use `text_proj` — must match Colab code or `load_state_dict` crashes |
| Streamlit Cloud: `torch==2.6.0+cpu` no wheels for Python 3.14 | Removed version pin — just `torch` in requirements.txt |
| Streamlit Cloud: `--extra-index-url` 403 with uv | Removed extra index URL — PyPI has CPU-compatible torch now |
| GitHub push: password auth rejected | Used Personal Access Token; then cleaned token from remote URL |
| BatchNorm crash with batch_size=1 | BatchNorm1d needs batch ≥ 2 in training mode; kept minimum batch=2 |

---

## Next steps to improve quality

1. **Train more epochs** (500+) — text conditioning sharpens gradually
2. **Super-resolution layer** — upscale 64×64 → 256×256 (reference repo had `sr_model.py`)
3. **Stronger text conditioning** — replace DistilBERT with CLIP text encoder
4. **Reduce mode collapse** — try spectral normalization on discriminator layers

---

## How to run locally

```powershell
# Inference
flower\Scripts\python.exe -m src.generate --prompt "a flower with pink petals" --n 8

# Streamlit demo
flower\Scripts\streamlit run app.py
```

Both commands must be run from `C:\Users\LENOVO\Downloads\flower_project\`.
`HF_HOME=D:\hf_cache` is set as a user env var — new terminals inherit it automatically.
