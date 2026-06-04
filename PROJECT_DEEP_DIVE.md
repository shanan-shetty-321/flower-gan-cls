# GAN-CLS + DistilBERT Flower Generator — Complete Deep Dive
### Interview Prep + Full Technical Understanding

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [Background theory — GANs](#2-background-theory--gans)
3. [Background theory — Conditional GANs & GAN-CLS](#3-background-theory--conditional-gans--gan-cls)
4. [Background theory — BERT & DistilBERT](#4-background-theory--bert--distilbert)
5. [Dataset — flowers-blip-captions](#5-dataset--flowers-blip-captions)
6. [Architecture deep dive](#6-architecture-deep-dive)
7. [Loss functions explained](#7-loss-functions-explained)
8. [Training pipeline end-to-end](#8-training-pipeline-end-to-end)
9. [Why each hyperparameter is what it is](#9-why-each-hyperparameter-is-what-it-is)
10. [DVC — what it is and how it applies here](#10-dvc--what-it-is-and-how-it-applies-here)
11. [Deployment architecture](#11-deployment-architecture)
12. [Problems encountered and how they were solved](#12-problems-encountered-and-how-they-were-solved)
13. [Interview Q&A — questions you will be asked](#13-interview-qa--questions-you-will-be-asked)
14. [What you'd do differently / improvements](#14-what-youd-do-differently--improvements)
15. [Glossary](#15-glossary)

---

## 1. What this project is

**One sentence:** Given a natural language description of a flower, generate a 64×64 pixel image of a flower that matches the description.

**Example:**
- Input: *"this flower has bright yellow petals with a dark center"*
- Output: A generated 64×64 RGB image of a yellow flower

**Why this problem is hard:**
- The model must understand language AND generate images — two completely different modalities
- It must link *specific words* to *visual features* (yellow → yellow pixels, dark center → dark center pixels)
- GANs are notoriously unstable to train — too easy for the discriminator → generator produces noise; too easy for generator → discriminator stops learning
- 6552 training samples is a relatively small dataset for a generative model

**What makes this different from a normal GAN:**
A normal GAN generates random images from noise. This GAN takes *text as input* and generates an image that matches that text. That's the "conditional" in GAN-CLS.

---

## 2. Background theory — GANs

### What is a GAN?

A Generative Adversarial Network (Goodfellow et al., 2014) is a framework with two neural networks competing:

```
Random noise (z) ──→ [ GENERATOR G ] ──→ Fake image
                                              │
Real images ──────────────────────────→ [ DISCRIMINATOR D ] ──→ Real (1) or Fake (0)?
```

- **Generator G**: takes random noise, outputs an image. Goal: fool D into saying "real"
- **Discriminator D**: takes an image, outputs a probability. Goal: correctly label real vs fake

They train adversarially (against each other). Think of it as a counterfeiter (G) vs a detective (D). As the detective gets better at spotting fakes, the counterfeiter gets better at making convincing fakes. Over time, the counterfeiter makes fakes so good the detective can't tell.

### GAN loss (original formulation)

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

In practice (implemented in code):
- D maximizes: `BCE(D(real), 1) + BCE(D(fake), 0)`
- G maximizes: `BCE(D(G(z)), 1)` (i.e., G tries to make D output 1 for its fake images)

### DCGAN (Deep Convolutional GAN)

The backbone used in this project. Key DCGAN rules:
- Replace pooling with strided convolutions
- Use BatchNorm in both G and D (except D's input layer and G's output layer)
- Use ReLU in G, LeakyReLU in D
- No fully connected layers (except for the initial projection in G)

DCGAN is stable and well-understood — the right choice for a 64×64 image generation task.

---

## 3. Background theory — Conditional GANs & GAN-CLS

### Conditional GAN (cGAN)

Instead of generating *any* image, a cGAN generates an image conditioned on some input `y`:

```
G(z, y) → image that matches condition y
D(x, y) → is this image real AND does it match condition y?
```

`y` can be a class label (digit 0-9), a class embedding, or — in our case — a **text embedding**.

### GAN-CLS (Matching-Aware Discriminator)

Reed et al. 2016 introduced a key insight: a normal cGAN discriminator only asks "is this image real?" — it doesn't actually verify text-image alignment.

**The problem:** The discriminator could learn to say "real" for any realistic-looking flower, regardless of whether the text matches. The generator then just learns to make realistic flowers and ignores the text.

**The GAN-CLS fix:** Train the discriminator on THREE types of pairs, not two:

| Pair type | Label | What it teaches |
|---|---|---|
| Real image + **matching** caption | 1 (real) | This is what a real, aligned pair looks like |
| Real image + **wrong** caption | 0 (fake) | Real flower + wrong text = reject |
| **Fake** image + matching caption | 0 (fake) | Generated flowers should look like their description |

The "wrong pair" term (real image + mismatched caption) is what forces the model to actually learn *text-image alignment*. Without it, the generator just ignores the text.

**In code:**
```python
# Discriminator sees 3 types of inputs:
real_p,  _ = D(real_images,  right_captions)   # should output ~1
wrong_p, _ = D(wrong_images, right_captions)   # should output ~0 (wrong pair)
fake_p,  _ = D(fake_images,  right_captions)   # should output ~0 (generated)

d_loss = BCE(real_p, 1) + BCE(wrong_p, 0) + BCE(fake_p, 0)
```

### Why it's called GAN-CLS

CLS = **C**onditional **L**atent **S**pace (matching-aware). The "matching-aware" part refers to the wrong-pair discrimination that forces the model to understand text-image relationships.

---

## 4. Background theory — BERT & DistilBERT

### What is BERT?

BERT (Bidirectional Encoder Representations from Transformers, Google 2018) is a language model pre-trained on billions of text documents. It turns words into vectors that capture *meaning*.

Key insight: "rose" and "petal" should have similar vector representations to "flower" because they appear in similar contexts in text. BERT learned this from reading the entire internet.

### What is a Transformer?

The architecture behind BERT. Key innovation: **self-attention** — every word in a sentence can "look at" every other word to understand context.

```
"this flower has YELLOW petals"
         ↑          ↑
YELLOW attends to "flower" and "petals" to understand it means flower color
```

BERT has 12 transformer layers, each refining the word representations based on context.

### What is DistilBERT?

DistilBERT = distilled BERT. It's a smaller, faster version of BERT that keeps 97% of BERT's language understanding with 40% fewer parameters.

| Model | Layers | Parameters | Speed |
|---|---|---|---|
| BERT-base | 12 | 110M | 1× |
| DistilBERT | 6 | 66M | 1.6× faster |

We use `distilbert-base-uncased` — uncased means all text is lowercased before encoding (so "Rose" and "rose" give the same embedding).

### Why we freeze DistilBERT

DistilBERT has 66 million parameters. The GAN has ~10 million. If we trained both simultaneously:
1. The gradients from the GAN loss would corrupt BERT's carefully learned language representations
2. Training would be extremely slow (66M extra parameters to update every step)
3. With only 6552 samples, we'd massively overfit

**Frozen = we treat DistilBERT as a fixed feature extractor.** We run it once at startup to convert all captions to embeddings, save those to RAM, and never touch DistilBERT again during training.

### Mean pooling

DistilBERT outputs a vector per token (word piece). A sentence of 10 tokens gets 10 vectors of size 768. We need ONE vector per sentence.

**Why mean pooling?** Average all token vectors (weighted by attention mask to skip padding). This is simple, robust, and standard.

```python
hidden = model(**enc).last_hidden_state      # (B, T, 768) — T vectors per sentence
mask   = enc["attention_mask"].unsqueeze(-1) # (B, T, 1) — 1 for real tokens, 0 for padding
pooled = (hidden * mask).sum(1) / mask.sum(1) # weighted average → (B, 768)
```

(Note: full BERT has a [CLS] token pooler head, but DistilBERT doesn't — mean pooling is the recommended approach for DistilBERT)

---

## 5. Dataset — flowers-blip-captions

### What is Oxford-102 Flowers?

A standard computer vision benchmark — 8,189 images of 102 flower species (roses, tulips, sunflowers, etc.). Each image is labelled with one of 102 class IDs.

### What are BLIP captions?

BLIP (Bootstrapping Language-Image Pre-training) is a vision-language model that can generate text descriptions of images. The `pranked03/flowers-blip-captions` dataset ran BLIP on every Oxford-102 flower image to auto-generate a natural language caption.

**Example caption:** "this flower has soft pink petals arranged in a circular pattern with a yellow center"

### Why this dataset is good for us

- Pre-labelled with class IDs (102 classes) — needed for the wrong-pair sampling (pick image from a *different* class)
- BLIP captions are descriptive and consistent in style
- Loads in one line via HuggingFace `datasets` — no manual download, no HDF5 wrangling
- 6552 training samples — small enough to preload into RAM (crucial for training speed)

### Wrong-pair sampling

The GAN-CLS trick requires a "wrong image" per sample — a real image from a *different flower class*. Implementation:

```python
def _wrong_idx(self, label):
    j = np.random.randint(self.n)
    while self.labels[j] == label:   # keep sampling until different class
        j = np.random.randint(self.n)
    return j
```

Simple but effective — random sampling from different classes guarantees the wrong pair is semantically different.

### Why we preload images into RAM

**Problem discovered during training:** The first run had 400 seconds per epoch instead of the expected ~12 seconds. Root cause: HuggingFace datasets are backed by Apache Parquet files. Every `__getitem__` call was decompressing and decoding a JPEG from disk — ~40ms per image × 6552 images = 260 seconds just in data loading, before any GPU work.

**Fix:** Load and transform all 6552 images into a single tensor at startup:

```python
self.images = torch.stack([transform(row["image"]) for row in hf_split])
# (6552, 3, 64, 64) float tensor sitting in RAM — ~100MB, instant access
```

After this fix: epochs dropped from 400s to 12.5s. The GPU was utilized >95% of the time instead of <5%.

**Lesson:** Always profile your DataLoader. A slow DataLoader is one of the most common training bottlenecks.

---

## 6. Architecture deep dive

### Generator

Takes: noise vector (100-d) + caption embedding (768-d)
Produces: 64×64 RGB image

```
caption (768-d)
    │
    ▼
Linear(768 → 128) + BatchNorm1d + LeakyReLU    ← project to compact conditioning vector
    │
    ▼
embed_out (128-d, shape: B×128×1×1)
    │
    ┤ concatenate along channel dim
    │
noise (100-d, shape: B×100×1×1)
    │
    ▼
Combined (228-d, shape: B×228×1×1)
    │
    ▼
ConvTranspose2d(228→512, k=4, s=1, p=0) + BN + ReLU   → B×512×4×4
ConvTranspose2d(512→256, k=4, s=2, p=1) + BN + ReLU   → B×256×8×8
ConvTranspose2d(256→128, k=4, s=2, p=1) + BN + ReLU   → B×128×16×16
ConvTranspose2d(128→64,  k=4, s=2, p=1) + BN + ReLU   → B×64×32×32
ConvTranspose2d(64→3,    k=4, s=2, p=1) + Tanh         → B×3×64×64
```

**Why ConvTranspose2d?** It's an "upsampling convolution" — the inverse of Conv2d. It doubles spatial dimensions at each step (stride=2). Also called a "deconvolution" (though that's technically a misnomer).

**Why Tanh at output?** Tanh outputs values in [-1, 1]. Images are normalized to [-1, 1] for training (pixel values 0-255 → normalized to [-1, 1]). Tanh allows the generator to output any value in this range.

**Why BatchNorm in Generator?** Stabilizes training — normalizes activations so gradients don't explode or vanish through the deep network.

**Why concatenate text with noise?** This is the conditioning mechanism. The text tells the generator "what kind of flower" and the noise provides "which specific instance" (variation). Different noise vectors with the same text embedding produce different but semantically similar flowers.

### Discriminator

Takes: 64×64 RGB image + caption embedding (768-d)
Produces: real/fake probability [0,1] + feature map (for feature matching loss)

```
Image (B×3×64×64)
    │
    ▼
Conv2d(3→64,   k=4, s=2, p=1) + LeakyReLU        → B×64×32×32   (no BN on first layer)
Conv2d(64→128, k=4, s=2, p=1) + BN + LeakyReLU   → B×128×16×16
Conv2d(128→256,k=4, s=2, p=1) + BN + LeakyReLU   → B×256×8×8
Conv2d(256→512,k=4, s=2, p=1) + BN + LeakyReLU   → B×512×4×4    ← feature map returned here
    │
    ├─── also returned as `feats` for feature matching loss
    │
caption (768-d)
    │
    ▼
Linear(768→128) + BatchNorm1d + LeakyReLU
    │
    ▼
reshape to (B×128×1×1) → tile/repeat → (B×128×4×4)
    │
    ┤ concatenate along channel dim
    │
Combined (B×640×4×4)   [512 image features + 128 text features]
    │
    ▼
Conv2d(640→1, k=4, s=1, p=0) + Sigmoid            → B×1×1×1 → scalar per sample
```

**Why tile the text embedding?** The image features are a spatial grid (4×4). To combine with text, we expand the text vector to match that spatial size — repeat the 128-d vector at every grid position. Then concatenate: every 4×4 location "sees" both local image features AND the full text description.

**Why no BatchNorm on first layer?** Standard DCGAN practice — normalizing the raw pixels can lose information about the overall image distribution.

**Why LeakyReLU in Discriminator?** Regular ReLU kills negative gradients entirely (dead neurons). LeakyReLU allows a small negative slope (0.2), keeping gradients flowing even for negative activations. This is especially important in the discriminator.

**Why return feature map?** The 512×4×4 feature map is used for the Generator's feature matching loss. See Loss Functions section.

---

## 7. Loss functions explained

### Discriminator loss

```python
d_loss = BCE(real_p, REAL_LABEL) + BCE(wrong_p, 0) + BCE(fake_p, 0)
```

Three terms:
1. `BCE(real_p, 0.9)` — penalize D for not recognizing real+matching pairs as real (label 0.9, not 1.0 — label smoothing)
2. `BCE(wrong_p, 0)` — penalize D for not recognizing real+wrong-text pairs as fake
3. `BCE(fake_p, 0)` — penalize D for not recognizing generated images as fake

**BCE (Binary Cross Entropy):** `BCE(p, y) = -(y·log(p) + (1-y)·log(1-p))`
- When y=1 (should be real): loss = -log(p). High loss when p is close to 0 (D says fake when it should say real)
- When y=0 (should be fake): loss = -log(1-p). High loss when p is close to 1 (D says real when it should say fake)

**Label smoothing (REAL_LABEL=0.9 not 1.0):** Using 0.9 instead of 1.0 as the "real" target prevents the discriminator from becoming overconfident. If D is too good, its gradients become nearly zero, and the generator receives no useful training signal. Smoothing keeps the training signal alive.

### Generator loss

```python
g_loss = BCE(fake_p, 1) + L1_COEFF * L1(fake, real) + L2_COEFF * L2(feat_fake.mean(0), feat_real.mean(0))
```

Three terms:

**1. Adversarial BCE:** `BCE(fake_p, 1)`
- Generator wants D to output 1 (real) for its fakes
- This is the core adversarial signal — "fool the discriminator"
- Uses label 1.0 (not smoothed) because we want G to fully convince D

**2. L1 Reconstruction Loss:** `10 × L1(fake_image, real_image)`
- Direct pixel-wise comparison between generated and real images
- Encourages generated images to actually look like real flowers (not just fool the discriminator)
- L1 = mean absolute difference: `mean(|fake - real|)` per pixel
- **Why L1 and not L2?** L1 is less sensitive to outliers (extreme pixel differences) and tends to produce sharper images. L2 tends to produce blurry results because it heavily penalizes large differences, making the model predict the "safe" average.
- **Why coefficient 10?** We tried 50 (original repo value) and found it caused blur + mode collapse. The high weight forced the generator to produce images that averaged all real flowers. Dropping to 10 balances reconstruction against adversarial creativity.

**3. Feature Matching L2:** `100 × L2(mean_feats_fake, mean_feats_real)`
- Compares the *discriminator's internal feature representations* of fake vs real images
- `feat = D(image)[1]` — the 512×4×4 feature map from the discriminator
- Mean across the batch: `feat.mean(0)` → (512, 4, 4) average feature map
- L2 = mean squared difference: `mean((fake_feats - real_feats)²)`
- `.detach()` on real features: we don't want gradients to flow back through D's real-image pass (we're only training G here)

**Why feature matching?** Pixel-level L1/L2 is a blunt instrument — it doesn't understand that a slightly shifted flower is actually similar to the target. Feature matching operates in a learned semantic space: if fake and real images produce similar discriminator features, they're structurally similar at a higher level. This produces more realistic textures and reduces training instability.

**Why coefficient 100?** Feature values are small (after BatchNorm), so L2 is naturally small. We upweight it to make it comparable in magnitude to the BCE loss.

### Summary of gradient flow

```
Real images + captions ──→ D ──→ d_loss (updates D only)
Wrong images + captions ──→ D ──┘

Noise + captions ──→ G ──→ fake ──→ D ──→ g_loss (updates G only)
                               ↑
                    real ──→ D ──→ feature matching (G loss, detach D)
```

Key: D's weights are frozen during G's update step (`fake.detach()` when computing D loss; optimizer_D and optimizer_G are separate).

---

## 8. Training pipeline end-to-end

### Step 1: Precompute embeddings (once)

```python
encoder = TextEncoder()  # load frozen DistilBERT
for batch_of_captions in dataset:
    embeddings.append(encoder.encode(batch_of_captions))
# Save to disk: models/text_embeddings.pt
```

**Why precompute?** DistilBERT is slow on CPU (and moderate on GPU). Running it every training step would dominate epoch time. Since it's frozen, the embeddings never change — compute once, cache, reuse.

### Step 2: Preload images (once)

```python
images = torch.stack([transform(row["image"]) for row in dataset])
# (6552, 3, 64, 64) in RAM
```

### Step 3: Training loop (per epoch)

For each batch:
```
1. Sample batch: real images, wrong images, embeddings
2. Generate fake images: G(noise, embeddings)
3. Update Discriminator:
   - Run D on real, wrong, fake → 3 probabilities
   - Compute d_loss = BCE(real,1) + BCE(wrong,0) + BCE(fake,0)
   - Backprop → update D weights
4. Generate NEW fake images (important: re-generate, don't reuse)
5. Update Generator:
   - Run D on fake → probability + feature map
   - Run D on real → feature map (for feature matching)
   - Compute g_loss = BCE(fake_p,1) + L1(fake,real) + L2(feats)
   - Backprop → update G weights only
6. Repeat for all batches
```

**Why generate new fakes in step 4?** If we reuse the fakes from step 3, they were generated with the old G (before D was updated). The G loss should reflect D's current state, not its previous state.

### Step 4: Checkpointing

- Every N epochs: save sample grid (fixed noise + fixed embeddings → comparable across epochs)
- When G loss is best: save `generator_best.pth`
- Every 50 epochs: save `generator_epoch_XXXX.pth` (in case best-by-loss isn't best-by-eye)

**Note on "best by loss" vs "best by eye":** In GANs, lowest loss ≠ best visual quality. The loss oscillates as G and D compete. Always check the sample grid images to judge quality.

### Adam optimizer with β=(0.5, 0.999)

Standard GAN training uses `betas=(0.5, 0.999)` instead of the default `(0.9, 0.999)`.

- β1=0.5 (instead of 0.9): reduces the exponential moving average of past gradients. In GANs, the loss landscape changes quickly because G and D keep updating against each other. A lower β1 means the optimizer responds more quickly to recent gradients. β1=0.9 can cause oscillations.
- β2=0.999: standard, keeps the second moment estimate stable.
- lr=0.0002: standard DCGAN learning rate.

---

## 9. Why each hyperparameter is what it is

| Hyperparameter | Value | Why |
|---|---|---|
| `noise_dim` | 100 | Standard DCGAN. Enough to encode variety, not so large it dominates the text conditioning |
| `embed_out_dim` | 128 | 768-d is too large to concatenate raw with 100-d noise. Project to 128 to balance the two. |
| `batch_size` | 64 | Larger batches → more stable gradients. BatchNorm needs reasonably large batches. 64 fits in T4 GPU RAM. |
| `lr` | 0.0002 | DCGAN paper recommendation. Higher → unstable. Lower → too slow. |
| `betas` | (0.5, 0.999) | DCGAN paper recommendation. β1=0.5 handles the changing GAN loss landscape. |
| `l1_coeff` | 10 | Tried 50 → blurry, mode collapse. 10 provides reconstruction guidance without dominating. |
| `l2_coeff` | 100 | Feature values are small after BN, so L2 is naturally tiny. ×100 makes it comparable to BCE. |
| `REAL_LABEL` | 0.9 | Label smoothing. Prevents D from being overconfident, keeps G gradients alive. |
| `epochs` | 450 | ~12.5s/epoch × 450 = ~94 min on T4. Quality improves gradually; 450 is a good cost/quality tradeoff. |
| `IMAGE_SIZE` | 64×64 | This architecture maxes out at 64 with 5 ConvTranspose layers. Higher res needs StackGAN or progressive growing. |

---

## 10. DVC — what it is and how it applies here

### What is DVC?

**DVC (Data Version Control)** is Git for data and ML models. Git tracks code changes, but it can't efficiently track large binary files (datasets, model weights). DVC solves this.

DVC stores large files in remote storage (S3, Google Drive, HuggingFace, etc.) and commits lightweight pointer files (`.dvc` files) to Git. When someone clones your repo, they get the pointers, then run `dvc pull` to download the actual data.

### Core DVC concepts

**1. Data versioning**
```bash
dvc add data/flowers_dataset/       # tracks the dataset
git add data/flowers_dataset.dvc    # commit the pointer
dvc push                            # upload dataset to remote storage
```

**2. Pipeline (dvc.yaml)**
```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps: [data/raw/, src/preprocess.py]
    outs: [data/processed/]
  
  embed:
    cmd: python src/embed_captions.py
    deps: [data/processed/, src/embed_captions.py]
    outs: [models/text_embeddings.pt]
  
  train:
    cmd: python -m src.train --config configs/train_config.yaml
    deps: [models/text_embeddings.pt, src/train.py, configs/train_config.yaml]
    outs: [models/generator_best.pth]
    metrics: [outputs/metrics.json]
```

**3. Experiment tracking**
```bash
dvc exp run            # run the pipeline
dvc exp show           # compare experiments
dvc params diff        # compare hyperparameters between runs
```

### How DVC would improve this project

| Without DVC (current) | With DVC |
|---|---|
| Dataset downloaded fresh every Colab run | `dvc pull` → cached, versioned |
| Embeddings recomputed every run | `dvc repro` only reruns changed stages |
| Can't compare experiment results | `dvc exp show` shows all runs with metrics |
| Model versions tracked manually | Every model linked to exact code + data version |
| No reproducibility guarantee | Anyone can `git checkout v1.0 && dvc checkout` to reproduce exactly |

### DVC in the context of this project

```bash
# Initialize
git init && dvc init

# Track dataset
dvc add data/flowers/         # ~500MB dataset
git add data/flowers.dvc .gitignore
git commit -m "add dataset pointer"

# Set up remote (Google Drive, S3, etc.)
dvc remote add myremote gdrive://YOUR_FOLDER_ID
dvc push

# Track embeddings (intermediate output)
dvc add models/text_embeddings.pt

# Track trained model
dvc add models/generator_best.pth

# Define pipeline
cat > dvc.yaml << EOF
stages:
  embed:
    cmd: python -m src.embed_captions
    deps: [src/text_encoder.py]
    outs: [models/text_embeddings.pt]
  train:
    cmd: python -m src.train
    deps: [models/text_embeddings.pt, src/train.py, configs/train_config.yaml]
    outs: [models/generator_best.pth]
    metrics: [outputs/metrics.json]
EOF

# Run + track experiment
dvc exp run --set-param train_config.yaml:l1_coeff=10
dvc exp run --set-param train_config.yaml:l1_coeff=50
dvc exp show   # compare both runs side by side
```

### Why we didn't use DVC in this project

1. **Simplicity:** For a portfolio project with a single training run, the overhead of DVC setup wasn't justified
2. **HuggingFace alternatives:** HF Hub already handles model versioning; HF Datasets handles data versioning
3. **Colab constraints:** DVC works best with persistent storage; Colab's ephemeral environment complicates it

**In a production/team ML project, DVC would be essential.**

---

## 11. Deployment architecture

### Local inference

```
User runs: python -m src.generate --prompt "yellow flower"
    │
    ├── TextEncoder loads DistilBERT from D:\hf_cache (already downloaded)
    ├── Generator loads weights from models/generator_best.pth
    ├── Encode prompt → 768-d embedding
    ├── Generate 8 random noise vectors
    ├── Forward pass: G(noise, embedding) → (8, 3, 64, 64)
    ├── Denormalize: [-1,1] → [0,1]
    └── Save grid to outputs/generated.png
```

### Streamlit app

```
User visits: https://flower-gan-cls.streamlit.app
    │
    ├── @st.cache_resource (runs ONCE, cached for session):
    │   ├── hf_hub_download("natsu321/flower-gan-cls", "generator_best.pth")
    │   │   └── Downloads 18MB model from HuggingFace Hub → local /models/
    │   ├── TextEncoder loads DistilBERT (~270MB, cached by HF)
    │   └── Generator loads weights
    │
    ├── User types prompt + clicks Generate
    ├── encoder.encode(prompt) → embedding
    ├── G(random_noise, embedding) → images
    └── Display in Streamlit columns
```

**@st.cache_resource:** Critical Streamlit decorator. Ensures models are loaded only ONCE per app session (not on every button click). Without it, every "Generate" would reload 270MB of weights.

### Streamlit Cloud infrastructure

- Runs on Linux (Python 3.14)
- No GPU (inference on CPU — fast enough for 64×64 GAN)
- 1GB RAM limit (DistilBERT + Generator + Streamlit = ~800MB)
- Requirements installed via `uv` (fast pip alternative)
- App auto-restarts when GitHub repo is updated (push to main → auto-redeploy)

### Why model weights live on HuggingFace Hub (not GitHub)

- GitHub has a 100MB file size limit; even a single model file can exceed this
- Git is terrible at storing binary files (no diffing, full file copied on every change)
- HuggingFace Hub is designed for this: versioned model storage, fast CDN, free
- `hf_hub_download()` handles caching automatically

---

## 12. Problems encountered and how they were solved

### Problem 1: Training speed — 400 seconds per epoch

**Symptom:** First training run: epoch 1 took 386 seconds on T4 GPU.

**Root cause:** `datasets` library loads data lazily from Parquet files. Each `__getitem__` call decompressed + decoded a JPEG image from disk. With 6552 samples and batch_size=64, that's 102 disk reads per epoch — each taking ~3.8 seconds per image (sequential I/O).

**Fix:** Preload all 6552 images into a `torch.Tensor` at dataset init time. After that, `__getitem__` is just `self.images[idx]` — a memory lookup taking nanoseconds.

**Result:** 400s/epoch → 12.5s/epoch. **32× speedup.**

**Lesson:** Profile your DataLoader before blaming the model. `nvidia-smi dmon` during training will show GPU utilization. If GPU is < 50%, you have a data loading bottleneck.

---

### Problem 2: C: drive full

**Symptom:** `load_dataset()` crashed with "Not enough space on disk."

**Root cause:** C: drive had only 0.67 GB free. HuggingFace was trying to cache a multi-GB dataset there.

**Fix:** Set `HF_HOME=D:\hf_cache` as a permanent user environment variable. All HuggingFace downloads (DistilBERT ~270MB, dataset ~3GB) go to D: (31GB free).

```powershell
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf_cache", "User")
```

---

### Problem 3: load_state_dict key mismatch

**Symptom:** After training on Colab and downloading the checkpoint, `G.load_state_dict(torch.load("generator_best.pth"))` crashed with:

```
RuntimeError: Missing keys: text_embedding.weight, text_embedding.bias
Unexpected keys: text_proj.weight, text_proj.bias
```

**Root cause:** The Colab code named the text projection layer `text_proj`. The local `src/model.py` named it `text_embedding`. PyTorch saves weights by layer name. The checkpoint keys didn't match the local model.

**Fix:** Renamed `text_embedding → text_proj` in local `src/model.py` to match Colab.

**Lesson:** Maintain a single source of truth for model architecture. The Colab code and local inference code must use identical layer names.

---

### Problem 4: Streamlit Cloud — torch install failure

**Symptom:** Streamlit Cloud showed:
```
torch==2.6.0+cpu has no wheels with a matching Python ABI tag
```

**Root cause:** Streamlit Cloud uses Python 3.14 (very new). The pinned `torch==2.6.0+cpu` has no wheels for Python 3.14. Additionally, `--extra-index-url https://download.pytorch.org/whl/cpu` was returning 403 Forbidden with Streamlit's `uv` installer.

**Fix:** Remove version pin and `--extra-index-url`. Just use `torch` in `requirements.txt` — PyPI now has CPU-compatible torch wheels for Python 3.14.

```diff
- --extra-index-url https://download.pytorch.org/whl/cpu
- torch==2.6.0+cpu
- torchvision==0.21.0+cpu
+ torch
+ torchvision
```

---

### Problem 5: Mode collapse (all 8 images identical)

**Symptom:** When generating 8 images with different noise but same text, all 8 images looked nearly identical.

**Root cause:** L1_COEFF=50 was too high. The L1 loss dominated the adversarial loss, forcing the generator to minimize pixel distance to real images. The "safe" strategy became outputting the average of all training flowers regardless of noise — minimizing average L1 error.

**Fix:** Lower L1_COEFF from 50 to 10. Also added label smoothing (REAL_LABEL=0.9) to prevent the discriminator from dominating.

---

### Problem 6: Blurry images

**Root cause:** Same as mode collapse — high L1 loss. L1 minimization produces blurry predictions because the model learns to output the mean across possible outputs (hedges its bets).

**Fix:** Lowering L1_COEFF to 10 allows the adversarial loss to dominate, which pushes toward sharper, more realistic images (GANs are known for sharper results than VAEs/pure reconstruction loss models).

---

## 13. Interview Q&A — questions you will be asked

### On GANs

**Q: What is mode collapse and how did you handle it?**
A: Mode collapse is when the generator learns to produce only a small subset of valid outputs (e.g., always generating the same average flower regardless of noise input). In my project, I observed mode collapse when L1_COEFF was set to 50 — all 8 generated images for different noise vectors were nearly identical. I addressed it by: (1) lowering L1_COEFF from 50 to 10, reducing the reconstruction pressure; (2) adding label smoothing (real labels = 0.9 instead of 1.0) to prevent the discriminator from becoming overconfident and killing generator gradients.

**Q: How do you evaluate a GAN? What metrics did you use?**
A: The standard metric is FID (Fréchet Inception Distance) — it compares the distribution of generated images vs real images in a feature space from an Inception network. Lower FID = more realistic and diverse images. In this project, I used visual inspection of sample grids saved every 10 epochs because FID requires generating thousands of images, which is expensive. In production, I'd compute FID periodically.

**Q: Why did you use DCGAN and not something like StyleGAN or diffusion?**
A: DCGAN is well-understood, stable, and appropriate for 64×64 images with a small dataset (6552 samples). StyleGAN2 produces stunning results but requires hundreds of thousands of images and days of GPU training — overkill for a portfolio project and this dataset size. Diffusion models (Stable Diffusion) are state-of-the-art for quality but require massive compute and pre-training on billions of images; they're not practical to train from scratch on Oxford-102. DCGAN was the right tool for this scope.

**Q: What is the role of BatchNorm in GAN training?**
A: In the Generator, BatchNorm normalizes layer inputs to have zero mean and unit variance, which prevents gradient vanishing/explosion through 5 deep ConvTranspose layers. In the Discriminator, BatchNorm stabilizes training but is intentionally omitted on the first layer (raw pixel input) to preserve the image's statistical properties. Without BatchNorm, GAN training is much less stable.

**Q: Why use LeakyReLU in the Discriminator but ReLU in the Generator?**
A: LeakyReLU allows a small negative slope (0.2) for negative inputs, preventing "dead neurons" where neurons stop contributing to gradients. The discriminator needs to propagate gradients back through many layers for the generator to learn — dead neurons in D would starve G of learning signal. The generator uses standard ReLU because its job is to produce clean, positive-valued feature maps, and ReLU's sparsity helps with this.

---

### On text-image alignment

**Q: How does the model learn to connect text descriptions to visual features?**
A: Through the GAN-CLS wrong-pair mechanism. The discriminator is trained to reject real images paired with mismatched captions. When the generator tries to fool the discriminator, it must produce images that not only look realistic but also match their text conditioning. The text embedding (from DistilBERT) is concatenated with the noise vector in the generator, and concatenated with the image features in the discriminator — both networks "see" the text at every decision point.

**Q: Why use DistilBERT instead of training your own text encoder?**
A: DistilBERT is pre-trained on billions of text documents and already understands semantic meaning, syntactic structure, and relationships between words. Training a text encoder from scratch on 6552 flower captions would produce terrible representations — not enough data. Reusing pre-trained BERT representations means the system starts with strong language understanding for free.

**Q: Why freeze DistilBERT?**
A: Three reasons: (1) stability — jointly training 66M BERT parameters + 10M GAN parameters on a small dataset would cause the GAN's adversarial loss to corrupt BERT's learned representations; (2) speed — updating 66M parameters per step is expensive; (3) data efficiency — with only 6552 samples, fine-tuning BERT would massively overfit.

---

### On architecture choices

**Q: Why 64×64 and not higher resolution?**
A: This DCGAN architecture uses 5 ConvTranspose2d layers each doubling spatial resolution, starting from a 4×4 seed: 4→8→16→32→64. Going to 128×128 requires adding a 6th layer and significantly more training time and GPU memory. For a portfolio project proving the GAN-CLS concept, 64×64 is sufficient. Higher resolution would require StackGAN (progressive refinement in stages) or ProGAN (progressive growing).

**Q: What is feature matching and why is it useful?**
A: Feature matching is a technique where the generator loss includes a term comparing the discriminator's intermediate feature representations of fake vs real images. Instead of only optimizing to fool the discriminator's final output, the generator also learns to produce images that look similar to real images in the discriminator's learned feature space. This stabilizes training because feature matching gradients are more informative and less "all-or-nothing" than the binary real/fake signal.

---

### On deployment

**Q: How does the model get loaded in production? How do you avoid loading it on every request?**
A: The model weights are hosted on HuggingFace Hub. The Streamlit app uses `@st.cache_resource` to load the model exactly once per app session — the first user request triggers the download and model loading (~2 min), and subsequent requests use the cached model object in memory. Without `@st.cache_resource`, every button click would reload 270MB of weights, making the app unusably slow.

**Q: What would you add for a production deployment?**
A: (1) Input validation and content filtering — ensure prompts are flower-related; (2) Request rate limiting — inference takes a few seconds on CPU; (3) Model versioning — ability to A/B test different checkpoints; (4) Monitoring — track latency, error rates, user prompts; (5) A GPU inference server (e.g., Replicate, Modal) for faster generation.

---

### On training

**Q: How did you handle training on a machine with no GPU?**
A: I developed and tested locally on CPU (Intel i5, no GPU) — running smoke tests with 2-4 samples to verify the pipeline worked. All actual training happened on Google Colab's free T4 GPU. This is a standard workflow: develop locally on CPU, train on cloud GPU. The code uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")` to work transparently in both environments.

**Q: Your training loss wasn't monotonically decreasing. Is that a problem?**
A: No — GAN losses are not expected to monotonically decrease. G and D are competing; when D gets better, G's loss goes up. When G gets better, D's loss goes up. A healthy GAN training shows both losses oscillating within a stable range, not diverging. The real signal of training success is visual inspection of generated samples — do the images get more realistic over epochs?

**Q: How many parameters does your model have?**
A: Generator: approximately 7.5M parameters. Discriminator: approximately 10M parameters. DistilBERT: 66M (frozen, not trained). Total trained parameters: ~17.5M.

---

## 14. What you'd do differently / improvements

### Immediate improvements

**1. Better text conditioning with CLIP**
CLIP (Contrastive Language-Image Pre-training, OpenAI) was trained specifically on image-text pairs — it knows what "yellow petals" looks like visually. DistilBERT was trained on text alone. CLIP embeddings would give much stronger text-image alignment.

**2. Super-resolution post-processing**
Add a second network (ESRGAN or a simple CNN upsampler) to upscale 64×64 → 256×256. The GAN generates structure at 64×64; the super-resolution network adds detail. This is the StackGAN approach.

**3. Spectral normalization in Discriminator**
Spectral normalization constrains the Lipschitz constant of D, preventing it from becoming too powerful. This reduces mode collapse and training instability without label smoothing.

**4. Progressive growing (ProGAN)**
Start training at 4×4, gradually increase to 8×8, 16×16, 32×32, 64×64. Each resolution is easy to learn; the model builds up complexity progressively. Results in much higher quality at the same or higher resolution.

### Production-grade improvements

**5. DVC for experiment tracking**
Track every experiment: hyperparameters, dataset version, model weights, metrics. Makes results reproducible and comparable.

**6. MLflow or Weights & Biases for monitoring**
Real-time loss curves, sample grids, FID scores logged to a dashboard during training.

**7. Larger dataset**
6552 images is small. Augmenting (flip, rotate, color jitter) effectively multiplies the dataset. Or use additional flower datasets (iNaturalist has millions of flower images).

**8. WGAN-GP (Wasserstein GAN with Gradient Penalty)**
More stable training than standard GAN with BCE loss. Replaces BCE with Wasserstein distance, adds gradient penalty term. Known to reduce mode collapse.

---

## 15. Glossary

| Term | Definition |
|---|---|
| **GAN** | Generative Adversarial Network — two networks (G and D) competing to generate realistic data |
| **DCGAN** | Deep Convolutional GAN — uses ConvTranspose2d in G and Conv2d in D |
| **GAN-CLS** | Matching-aware conditional GAN — trains D on wrong pairs to enforce text-image alignment |
| **Conditional GAN** | GAN conditioned on extra information (class label, text) to control generation |
| **Mode collapse** | Generator produces only a few repeated outputs instead of diverse samples |
| **Label smoothing** | Using 0.9 instead of 1.0 as "real" target — prevents overconfidence |
| **Feature matching** | Generator loss term comparing discriminator intermediate features (fake vs real) |
| **BERT** | Bidirectional Encoder Representations from Transformers — large pre-trained language model |
| **DistilBERT** | Smaller, faster BERT (40% fewer parameters, 97% of performance) |
| **Transformer** | Neural network architecture using self-attention to process sequences |
| **Self-attention** | Mechanism allowing each token to attend to all other tokens in the sequence |
| **Mean pooling** | Average all token embeddings to get a single sentence embedding |
| **BatchNorm** | Normalizes layer inputs to zero mean/unit variance — stabilizes training |
| **LeakyReLU** | Activation function allowing small negative slope — prevents dead neurons |
| **ConvTranspose2d** | Upsampling convolution — increases spatial dimensions |
| **BCE** | Binary Cross Entropy — standard loss for binary classification (real/fake) |
| **L1 loss** | Mean absolute difference — robust to outliers, encourages sharpness |
| **L2 loss / MSE** | Mean squared difference — heavily penalizes large errors |
| **Embedding** | Dense vector representation of discrete data (words, images) |
| **Frozen model** | Model whose weights are fixed and not updated during training |
| **DVC** | Data Version Control — Git for data and ML models |
| **FID** | Fréchet Inception Distance — measures quality/diversity of generated images |
| **BLIP** | Bootstrapping Language-Image Pre-training — model that generates image captions |
| **HuggingFace Hub** | Platform for hosting pre-trained models and datasets |
| **Streamlit** | Python library for building ML demo web apps |
| **@st.cache_resource** | Streamlit decorator to load resources (models) once per session |
| **Parquet** | Columnar storage format used by HuggingFace datasets — efficient but slow for random access |
| **Denormalization** | Converting normalized tensor values [-1,1] back to image values [0,255] |
| **weights_init** | DCGAN initialization: conv ~N(0,0.02), BN ~N(1,0.02) |
| **wrong pair** | Real image paired with mismatched text caption — used to train D for alignment |
| **Adam** | Adaptive moment estimation optimizer — standard for deep learning |
| **β1, β2** | Adam momentum coefficients. GAN training uses β1=0.5 instead of default 0.9 |
