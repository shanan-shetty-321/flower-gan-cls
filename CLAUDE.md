# CLAUDE.md — Handoff & Project State

This file lets any Claude model (the user is on **Sonnet 4.6 1M** going forward;
this was scaffolded on Opus 4.8) continue seamlessly. Read it fully before acting.

## What this project is
Text-to-image synthesis: generate 64×64 flower images from text, using a
**GAN-CLS** conditional GAN conditioned on **frozen DistilBERT** caption
embeddings. Trained with **BCE + L1 + L2** losses. It reproduces, in a clean
rewrite, the approach of `github.com/nhanth301/Text2Image-GANCLS-BERT` (the only
one of 8 reference repos on a modern, non-TensorFlow stack).

## The user
- Email shanan@propheus.com. Building this as a portfolio/learning project.
- Wants to **understand** the code (chose "clean guided rewrite"), not just run it.
- Communicates casually; be concrete and teacherly, explain the *why*.

## Hard constraints (DO NOT FORGET)
- **No GPU.** Machine is Intel i5-10310U, 16 GB RAM, Intel UHD only — no CUDA.
  → Local = development, smoke-testing, and inference/demo ONLY.
  → **Training happens on Kaggle** (free T4/P100). User confirmed this choice.
- Windows 11, **PowerShell**. Python via `py -3.12`. Two Pythons: 3.13.6 + 3.12.9.
- The venv is **`flower/`** (Python 3.12.9). Call it directly — no need to activate:
  `flower\Scripts\python.exe ...`  (CPU-only torch 2.12.0+cpu installed).

## Decisions already made
| Decision | Choice |
|---|---|
| Training venue | **Kaggle** (free GPU); local is CPU dev/inference only |
| Build style | **Clean guided rewrite** (not a direct port of nhanth301) |
| Base architecture | DCGAN GAN-CLS @ 64×64, DistilBERT (768-d, frozen, mean-pooled) |
| Dataset | `pranked03/flowers-blip-captions` (HF datasets) |
| Stack | PyTorch + transformers + datasets + streamlit (no TensorFlow) |

## Architecture specifics (so behavior is reproducible)
- Generator: caption(768)→Linear→`embed_out_dim`(128) ⊕ noise(100) → 5 ConvTranspose
  layers (512→256→128→64→3), Tanh. Output (B,3,64,64) in [-1,1].
- Discriminator: 4 conv blocks → (B,512,4,4); caption→128 tiled over 4×4, concat,
  final conv+Sigmoid → scalar. Returns `(prob, features)`; features feed G's L2.
- Losses (`src/train.py`): D = BCE(real,1)+BCE(wrong,0)+BCE(fake,0);
  G = BCE(fake,1) + 50·L1(fake,real) + 100·L2(mean feat fake vs real).
- Config in `configs/train_config.yaml`. Embeddings precomputed once and cached
  to `models/text_embeddings.pt`.

## Status / what's done
- [x] Reviewed all 8 reference repos; chose + documented the approach.
- [x] Created `flower/` venv (Py 3.12), installing CPU torch 2.12.0 + HF stack.
- [x] Wrote full clean implementation: text_encoder, dataset, model, utils,
      train, generate, app.py, smoke_test.py.
- [x] Wrote configs, requirements.txt, .gitignore, README.md, Kaggle notebook.
- [ ] **NEXT: run `smoke_test.py`** to verify the pipeline end-to-end on CPU.
      (First run downloads DistilBERT ~270 MB + the dataset — needs internet.)
- [ ] Then guide the user through Kaggle upload + training.
- [ ] Then wire the downloaded checkpoint into the local demo.

## How to verify locally (fast)
```powershell
flower\Scripts\python.exe smoke_test.py
```
Expect: embeddings (16,768) OK, batch (4,3,64,64) OK, 2 train steps print D/G
losses, "SMOKE TEST PASSED". If it passes, the code is correct — quality is then
purely a function of GPU training epochs on Kaggle.

## Likely gotchas to watch
- `BatchNorm1d` in the text projection needs batch_size ≥ 2 (smoke uses 4 — fine).
- **C: drive has ~0.7 GB free. HF_HOME is permanently set to `D:\hf_cache`** (31 GB free).
  All HuggingFace downloads (DistilBERT + dataset) go to D:. Do not change this.
  When running any python command in a new shell, the env var is already set for the user.
- numpy 2.x is fine with torch 2.12; don't pin it down unnecessarily.
- On Kaggle DO NOT reinstall torch (it ships with CUDA) — only `transformers datasets pyyaml`.
- `embed_out_dim` must match between training and `src/generate.py`/`app.py` (default 128).
- If running commands manually in PowerShell, ensure `$env:HF_HOME = "D:\hf_cache"` is set
  (it's a user env var so new terminals inherit it automatically).

## Pointers
- Reference repo (exact match): https://github.com/nhanth301/Text2Image-GANCLS-BERT
- GAN-CLS paper: Reed et al. 2016, "Generative Adversarial Text-to-Image Synthesis".
