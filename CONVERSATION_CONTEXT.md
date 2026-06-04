# Project Conversation Context — GAN-CLS + DistilBERT Flower Generator

Send this file to any AI agent to resume with full context. It contains every decision, every file, and all code.

---

## Who is the user

- **Name:** Shanan Shetty | **Email:** shanan@propheus.com
- **GitHub:** https://github.com/shanan-shetty-321
- **HuggingFace:** https://huggingface.co/natsu321
- **Goal:** Portfolio/learning project — wants to understand the code, not just run it
- **Style:** Casual. Explain the *why*, not just the *how*

---

## Machine specs

| Property | Value |
|---|---|
| OS | Windows 11 Pro, PowerShell |
| CPU | Intel Core i5-10310U @ 1.70GHz |
| RAM | 16 GB |
| GPU | Intel UHD only — **NO NVIDIA / NO CUDA** |
| Python | 3.12.9 in `flower\` venv + system 3.13.6 |
| C: drive | ~0.7 GB free (critically low) |
| D: drive | ~31 GB free |

`HF_HOME=D:\hf_cache` set permanently as user env var — all HuggingFace downloads go to D:.
Venv: call directly as `flower\Scripts\python.exe` from project root (no activation needed).

---

## Project locations

- **Local:** `C:\Users\LENOVO\Downloads\flower_project\`
- **GitHub:** https://github.com/shanan-shetty-321/flower-gan-cls
- **HuggingFace model:** https://huggingface.co/natsu321/flower-gan-cls (`generator_best.pth`, 18 MB)
- **Live demo:** https://flower-gan-cls-ly29pxv4cappotnbqflhulp.streamlit.app/

---

## What was built

Text-to-Image Synthesis — type a flower description, get a 64×64 flower image.

### Architecture

- **Text encoder:** Frozen `distilbert-base-uncased`, mean-pooled → **768-d embedding**
- **Generator (DCGAN):** caption(768) → Linear+BN+LReLU → 128-d ⊕ noise(100) → 5× ConvTranspose2d → **64×64 RGB** (Tanh, range [-1,1])
- **Discriminator:** 64×64 → 4× Conv2d (64→128→256→512) → 4×4×512 ⊕ caption(128) tiled → Conv2d → Sigmoid → scalar + feature map
- **GAN-CLS losses:**
  - D = `BCE(real+right_text, 0.9)` + `BCE(real+wrong_text, 0)` + `BCE(fake, 0)`
  - G = `BCE(fake, 1)` + `10·L1(fake,real)` + `100·L2(feat_fake.mean, feat_real.mean)`
- **Dataset:** `pranked03/flowers-blip-captions` — Oxford-102 flowers + BLIP captions, 6552 samples
- **Trained:** 450 epochs, Google Colab T4 GPU, ~94 minutes

### Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Training venue | Google Colab T4 | No local GPU |
| Build style | Clean guided rewrite | Portfolio/learning |
| Reference | `nhanth301/Text2Image-GANCLS-BERT` | Only modern (non-TF) match |
| L1_COEFF | 10 (not 50) | 50 caused blur + mode collapse |
| REAL_LABEL | 0.9 | Label smoothing, prevents D dominating |
| Layer name | `text_proj` (not `text_embedding`) | Must match between Colab + local or load_state_dict crashes |
| Model hosting | HuggingFace Hub | Standard for ML demos |
| Deployment | Streamlit Community Cloud | Free, GitHub-connected |

---

## File structure

```
flower_project/
├── app.py                      Streamlit demo (auto-downloads model from HF Hub)
├── smoke_test.py               Local CPU pipeline sanity check
├── requirements.txt            No torch version pin (Streamlit Cloud Python 3.14)
├── README.md
├── CLAUDE.md                   AI handoff doc
├── CONVERSATION_CONTEXT.md     This file
├── .gitignore
├── configs/train_config.yaml
├── src/
│   ├── __init__.py
│   ├── text_encoder.py
│   ├── dataset.py
│   ├── model.py
│   ├── utils.py
│   ├── train.py
│   └── generate.py
├── kaggle/kaggle_train.ipynb
├── models/generator_best.pth   (gitignored, on HF Hub)
└── flower/                     (gitignored venv)
```

---

## LOCAL CODE — src/text_encoder.py

```python
import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

EMBED_DIM = 768  # DistilBERT hidden size

class TextEncoder:
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cpu"):
        self.device = device
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)  # frozen — never backprop into BERT

    @torch.no_grad()
    def encode(self, texts, max_length: int = 64) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        last_hidden = self.model(**enc).last_hidden_state   # (B, T, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)            # (B, 768)
        counts = mask.sum(dim=1).clamp(min=1e-9)            # (B, 1)
        return (summed / counts).cpu()                      # mean-pooled
```

---

## LOCAL CODE — src/model.py

```python
import torch
import torch.nn as nn
from .text_encoder import EMBED_DIM

class Generator(nn.Module):
    def __init__(self, channels: int = 3, noise_dim: int = 100,
                 embed_dim: int = EMBED_DIM, embed_out_dim: int = 128):
        super().__init__()
        self.noise_dim = noise_dim
        self.text_proj = nn.Sequential(   # NOTE: must be named text_proj
            nn.Linear(embed_dim, embed_out_dim),
            nn.BatchNorm1d(embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        def block(in_c, out_c, k=4, s=2, p=1, last=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False)]
            layers += [nn.Tanh()] if last else [nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            return layers
        self.model = nn.Sequential(
            *block(noise_dim + embed_out_dim, 512, k=4, s=1, p=0),  # -> 4x4
            *block(512, 256), *block(256, 128), *block(128, 64),
            *block(64, channels, last=True),                         # -> 64x64
        )

    def forward(self, noise, text):
        t = self.text_proj(text)
        t = t.view(t.size(0), t.size(1), 1, 1)
        z = torch.cat([t, noise], dim=1)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, channels: int = 3,
                 embed_dim: int = EMBED_DIM, embed_out_dim: int = 128):
        super().__init__()
        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if normalize: layers += [nn.BatchNorm2d(out_c)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers
        self.features = nn.Sequential(
            *block(channels, 64, normalize=False),
            *block(64, 128), *block(128, 256), *block(256, 512),
        )
        self.text_proj = nn.Sequential(   # NOTE: must be named text_proj
            nn.Linear(embed_dim, embed_out_dim),
            nn.BatchNorm1d(embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512 + embed_out_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, image, text):
        feats = self.features(image)
        t = self.text_proj(text)
        t = t.view(t.size(0), t.size(1), 1, 1)
        t = t.repeat(1, 1, feats.size(2), feats.size(3))
        x = torch.cat([feats, t], dim=1)
        prob = self.classifier(x).view(-1)
        return prob, feats  # feats used for feature-matching loss
```

---

## LOCAL CODE — src/dataset.py

```python
import numpy as np
import torch
from torch.utils.data import Dataset

class Text2ImageDataset(Dataset):
    def __init__(self, hf_split, embeddings: torch.Tensor, transform=None):
        self.ds = hf_split
        self.embeddings = embeddings
        self.transform = transform
        self.labels = list(hf_split["label"])
        self.n = len(self.labels)

    def __len__(self): return self.n

    def _to_tensor(self, img):
        if img.mode != "RGB": img = img.convert("RGB")
        return self.transform(img)

    def _sample_wrong_index(self, label):
        j = np.random.randint(self.n)
        while self.labels[j] == label:
            j = np.random.randint(self.n)
        return j

    def __getitem__(self, idx):
        row = self.ds[idx]
        right_image = self._to_tensor(row["image"])
        right_embed = self.embeddings[idx]
        wrong_idx = self._sample_wrong_index(row["label"])
        wrong_image = self._to_tensor(self.ds[wrong_idx]["image"])
        return {
            "right_images": right_image,
            "right_embed": right_embed,
            "wrong_images": wrong_image,
            "txt": str(row["text"]),
        }
```

---

## LOCAL CODE — src/utils.py

```python
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils

IMAGE_SIZE = 64

transform = T.Compose([
    T.Resize(IMAGE_SIZE), T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
])

def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def save_checkpoint(state_dict, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state_dict, path)

def denorm(t):
    return (t.clamp(-1, 1) + 1) / 2

def save_sample_grid(images, path, nrow=4):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    vutils.save_image(denorm(images), path, nrow=nrow)
```

---

## LOCAL CODE — src/train.py

```python
import argparse, os, time
import torch, torch.nn as nn, yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from .dataset import Text2ImageDataset
from .model import Generator, Discriminator
from .text_encoder import TextEncoder, EMBED_DIM
from .utils import transform, weights_init, save_checkpoint, save_sample_grid

def precompute_embeddings(texts, encoder, cache_path, batch_size=64):
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path)
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunks.append(encoder.encode(texts[i:i+batch_size]))
    emb = torch.cat(chunks, dim=0)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(emb, cache_path)
    return emb

def train(config_path):
    with open(config_path) as f: cfg = yaml.safe_load(f)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset(cfg["dataset"], split="train")
    encoder = TextEncoder(cfg["text_model"], device=str(device))
    embeddings = precompute_embeddings(ds["text"], encoder, cfg.get("embeddings_cache"), cfg["batch_size"])
    dataset = Text2ImageDataset(ds, embeddings, transform)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], drop_last=True)
    G = Generator(channels=3, noise_dim=cfg["noise_dim"],
                  embed_dim=EMBED_DIM, embed_out_dim=cfg["embed_out_dim"]).to(device)
    D = Discriminator(channels=3, embed_dim=EMBED_DIM,
                      embed_out_dim=cfg["embed_out_dim"]).to(device)
    G.apply(weights_init); D.apply(weights_init)
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["learning_rate"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["learning_rate"], betas=(0.5, 0.999))
    bce, l1, l2 = nn.BCELoss(), nn.L1Loss(), nn.MSELoss()
    noise_dim = cfg["noise_dim"]
    fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)
    fixed_embed = embeddings[:16].to(device)
    os.makedirs("models", exist_ok=True); os.makedirs("outputs/samples", exist_ok=True)

    for epoch in range(1, cfg["epochs"]+1):
        t0 = time.time(); d_run = g_run = 0.0
        for batch in loader:
            real = batch["right_images"].to(device)
            wrong = batch["wrong_images"].to(device)
            emb = batch["right_embed"].to(device)
            bs = real.size(0)
            ones = torch.ones(bs, device=device); zeros = torch.zeros(bs, device=device)
            opt_D.zero_grad()
            fake = G(torch.randn(bs, noise_dim, 1, 1, device=device), emb)
            real_p, _ = D(real, emb); wrong_p, _ = D(wrong, emb); fake_p, _ = D(fake.detach(), emb)
            d_loss = bce(real_p, ones) + bce(wrong_p, zeros) + bce(fake_p, zeros)
            d_loss.backward(); opt_D.step()
            opt_G.zero_grad()
            fake = G(torch.randn(bs, noise_dim, 1, 1, device=device), emb)
            fake_p, ff = D(fake, emb); _, rf = D(real, emb)
            g_loss = bce(fake_p, ones) + cfg["l1_coeff"]*l1(fake, real) + cfg["l2_coeff"]*l2(ff.mean(0), rf.mean(0).detach())
            g_loss.backward(); opt_G.step()
            d_run += d_loss.item(); g_run += g_loss.item()
        n = len(loader)
        print(f"epoch {epoch}/{cfg['epochs']}  D {d_run/n:.3f}  G {g_run/n:.3f}  ({time.time()-t0:.1f}s)")
        if epoch % cfg["sample_every"] == 0 or epoch == 1:
            G.eval()
            with torch.no_grad(): save_sample_grid(G(fixed_noise, fixed_embed), f"outputs/samples/epoch_{epoch:04d}.png")
            G.train()
        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            save_checkpoint(G.state_dict(), f"models/generator_epoch_{epoch:04d}.pth")
            save_checkpoint(G.state_dict(), "models/generator_latest.pth")
            save_checkpoint(D.state_dict(), "models/discriminator_latest.pth")
    print("Training complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    train(ap.parse_args().config)
```

---

## LOCAL CODE — src/generate.py

```python
import argparse, torch
from .model import Generator
from .text_encoder import TextEncoder, EMBED_DIM
from .utils import save_sample_grid

def load_generator(ckpt_path, device, noise_dim=100, embed_out_dim=128):
    G = Generator(channels=3, noise_dim=noise_dim, embed_dim=EMBED_DIM, embed_out_dim=embed_out_dim).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.eval()
    return G

@torch.no_grad()
def generate(prompt, ckpt="models/generator_best.pth", n=8,
             noise_dim=100, embed_out_dim=128, out="outputs/generated.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TextEncoder(device=str(device))
    G = load_generator(ckpt, device, noise_dim, embed_out_dim)
    emb = encoder.encode([prompt]).to(device).repeat(n, 1)
    noise = torch.randn(n, noise_dim, 1, 1, device=device)
    images = G(noise, emb)
    save_sample_grid(images, out, nrow=min(n, 4))
    print(f"Saved {n} images for prompt: {prompt!r} -> {out}")
    return images

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ckpt", default="models/generator_best.pth")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--embed_out_dim", type=int, default=128)
    ap.add_argument("--out", default="outputs/generated.png")
    a = ap.parse_args()
    generate(a.prompt, a.ckpt, a.n, embed_out_dim=a.embed_out_dim, out=a.out)
```

---

## LOCAL CODE — app.py (Streamlit demo)

```python
import os, torch, streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
from src.model import Generator
from src.text_encoder import TextEncoder, EMBED_DIM
from src.utils import denorm

HF_REPO    = "natsu321/flower-gan-cls"
MODEL_FILE = "generator_best.pth"
EMBED_OUT  = 128
NOISE_DIM  = 100
LOCAL_PATH = os.path.join("models", MODEL_FILE)

def get_model_path():
    if os.path.exists(LOCAL_PATH): return LOCAL_PATH
    os.makedirs("models", exist_ok=True)
    return hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE, local_dir="models")

@st.cache_resource
def load_models():
    ckpt = get_model_path()
    encoder = TextEncoder(device="cpu")
    G = Generator(channels=3, noise_dim=NOISE_DIM, embed_dim=EMBED_DIM, embed_out_dim=EMBED_OUT)
    G.load_state_dict(torch.load(ckpt, map_location="cpu"))
    G.eval()
    return encoder, G

st.set_page_config(page_title="Flower GAN", page_icon="🌸")
st.title("🌸 Text-to-Flower Generator")
st.caption("GAN-CLS + DistilBERT · Trained on Oxford-102 flowers · 64×64")

prompt = st.text_input("Describe a flower", placeholder="e.g. this flower has soft pink petals and a yellow center")
n = st.slider("Number of images", 1, 8, 4)

if st.button("Generate", type="primary"):
    if not prompt.strip(): st.warning("Please enter a description."); st.stop()
    with st.spinner("Loading model..."): encoder, G = load_models()
    with st.spinner("Generating..."):
        with torch.no_grad():
            emb  = encoder.encode([prompt]).repeat(n, 1)
            imgs = denorm(G(torch.randn(n, NOISE_DIM, 1, 1), emb)).permute(0,2,3,1).numpy()
    cols = st.columns(min(n, 4))
    for i in range(n):
        cols[i % len(cols)].image(Image.fromarray((imgs[i]*255).astype("uint8")).resize((192,192), Image.NEAREST), caption=f"#{i+1}")
    st.caption(f'Prompt: "{prompt}"')
```

---

## LOCAL CODE — configs/train_config.yaml

```yaml
dataset: pranked03/flowers-blip-captions
text_model: distilbert-base-uncased
noise_dim: 100
embed_out_dim: 128
batch_size: 64
learning_rate: 0.0002
epochs: 200
l1_coeff: 50
l2_coeff: 100
num_workers: 2
sample_every: 5
save_every: 25
seed: 42
embeddings_cache: models/text_embeddings.pt
```

---

## LOCAL CODE — requirements.txt

```
torch
torchvision
transformers>=4.44
datasets>=2.20
huggingface_hub>=0.24
pyyaml>=6.0
pillow>=10.0
numpy>=1.26
streamlit>=1.38
```

Note: No torch version pin — Streamlit Cloud uses Python 3.14, pinning `torch==2.6.0+cpu` caused 403/no-wheels errors with uv installer.

---

## COLAB CODE — All 9 cells (final working version)

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

### Cell 5 — Dataset (images preloaded into RAM — critical for speed)
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
        self.images = torch.stack(self.images)  # (N,3,64,64) in RAM
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

print("Encoding captions with DistilBERT...")
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

os.makedirs("samples", exist_ok=True); os.makedirs("models", exist_ok=True)
fixed_noise = torch.randn(16, NOISE_DIM, 1, 1, device=DEVICE)
fixed_embed = embeddings[:16].to(DEVICE)
best_g_loss = float("inf")

for epoch in range(1, EPOCHS+1):
    t0 = time.time(); d_sum = g_sum = 0
    for batch in loader:
        real  = batch["right_images"].to(DEVICE)
        wrong = batch["wrong_images"].to(DEVICE)
        emb   = batch["right_embed"].to(DEVICE)
        bs    = real.size(0)
        ones  = torch.full((bs,), REAL_LABEL, device=DEVICE)
        zeros = torch.zeros(bs, device=DEVICE)

        opt_D.zero_grad()
        fake        = G(torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE), emb)
        real_p,  _  = D(real, emb); wrong_p, _ = D(wrong, emb); fake_p, _ = D(fake.detach(), emb)
        d_loss = bce(real_p, ones) + bce(wrong_p, zeros) + bce(fake_p, zeros)
        d_loss.backward(); opt_D.step()

        opt_G.zero_grad()
        fake        = G(torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE), emb)
        fake_p, f_f = D(fake, emb); _, f_r = D(real, emb)
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
    print(f"[{epoch:03d}/{EPOCHS}] D={avg_d:.3f}  G={avg_g:.3f}  ({'BEST ' if is_best else '      '}){time.time()-t0:.1f}s")

    if epoch % SAMPLE_EVERY == 0 or epoch == 1:
        G.eval()
        with torch.no_grad(): vutils.save_image(denorm(G(fixed_noise, fixed_embed)), f"samples/epoch_{epoch:04d}.png", nrow=4)
        G.train()
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        torch.save(G.state_dict(), f"models/generator_epoch_{epoch:04d}.pth")
        torch.save(G.state_dict(), "models/generator_latest.pth")
        torch.save(D.state_dict(), "models/discriminator_latest.pth")

print(f"Done! Best G loss: {best_g_loss:.3f}")
print("Download models/generator_best.pth from the Files panel (left sidebar).")
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
| First epoch took 400s (not ~12s) | DataLoader decoded images from parquet on every `__getitem__`. Fixed by preloading all 6552 images into a RAM tensor in `__init__`. |
| C: drive full (0.7 GB) | Set `HF_HOME=D:\hf_cache` permanently as user env var |
| `generator_latest.pth` not found | Changed default ckpt to `generator_best.pth` in `generate.py` and `app.py` |
| `text_embedding` vs `text_proj` key mismatch | Fixed `src/model.py` to use `text_proj` — must match Colab or `load_state_dict` crashes |
| Streamlit Cloud: no wheels for torch on Python 3.14 | Removed version pin — just `torch` in requirements.txt |
| Streamlit Cloud: `--extra-index-url` 403 with uv | Removed extra index URL entirely |
| GitHub push auth rejected | Used Personal Access Token; then cleaned token from remote URL |
| BatchNorm crash with batch_size=1 | BatchNorm1d needs batch ≥ 2 in training mode |
| HF warnings (`UNEXPECTED keys`) | Normal — unused MLM head weights from pretrained DistilBERT, safe to ignore |

---

## How to run locally

```powershell
# Inference
flower\Scripts\python.exe -m src.generate --prompt "a flower with pink petals" --n 8

# Streamlit demo
flower\Scripts\streamlit run app.py
```

Run from `C:\Users\LENOVO\Downloads\flower_project\`. `HF_HOME=D:\hf_cache` auto-inherited by new terminals.

## Next steps to improve

1. Train more epochs (500+) on Colab
2. Super-resolution layer: upscale 64×64 → 256×256
3. Replace DistilBERT with CLIP text encoder for stronger conditioning
4. Spectral normalization on discriminator to reduce mode collapse
