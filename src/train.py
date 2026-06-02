"""
GAN-CLS + DistilBERT training loop.

Run from the project root:
    python -m src.train --config configs/train_config.yaml

Pipeline:
  1. Load the flowers-blip-captions dataset.
  2. Precompute (and cache) DistilBERT caption embeddings once.
  3. Train generator + discriminator with the GAN-CLS losses:
       D = BCE(real,1) + BCE(wrong-pair,0) + BCE(fake,0)
       G = BCE(fake,1) + l1_coeff*L1(fake,real) + l2_coeff*L2(featmatch)
  4. Periodically save sample grids and generator checkpoints.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader

from .dataset import Text2ImageDataset
from .model import Generator, Discriminator
from .text_encoder import TextEncoder, EMBED_DIM
from .utils import transform, weights_init, save_checkpoint, save_sample_grid


def precompute_embeddings(texts, encoder, cache_path, batch_size=64):
    if cache_path and os.path.exists(cache_path):
        print(f"[embeddings] loading cache: {cache_path}")
        return torch.load(cache_path)
    print(f"[embeddings] encoding {len(texts)} captions with DistilBERT...")
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunks.append(encoder.encode(texts[i:i + batch_size]))
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{len(texts)}")
    emb = torch.cat(chunks, dim=0)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(emb, cache_path)
        print(f"[embeddings] cached -> {cache_path}")
    return emb


def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # --- data + embeddings ---
    ds = load_dataset(cfg["dataset"], split="train")
    encoder = TextEncoder(cfg["text_model"], device=str(device))
    embeddings = precompute_embeddings(
        ds["text"], encoder, cfg.get("embeddings_cache"), cfg["batch_size"]
    )
    dataset = Text2ImageDataset(ds, embeddings, transform)
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], drop_last=True,
    )

    # --- models ---
    G = Generator(channels=3, noise_dim=cfg["noise_dim"],
                  embed_dim=EMBED_DIM, embed_out_dim=cfg["embed_out_dim"]).to(device)
    D = Discriminator(channels=3, embed_dim=EMBED_DIM,
                      embed_out_dim=cfg["embed_out_dim"]).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["learning_rate"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["learning_rate"], betas=(0.5, 0.999))

    bce, l1, l2 = nn.BCELoss(), nn.L1Loss(), nn.MSELoss()
    noise_dim = cfg["noise_dim"]

    # Fixed noise + a few real embeddings so sample grids are comparable across epochs.
    fixed_noise = torch.randn(16, noise_dim, 1, 1, device=device)
    fixed_embed = embeddings[:16].to(device)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/samples", exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        d_running, g_running = 0.0, 0.0
        for batch in loader:
            real = batch["right_images"].to(device)
            wrong = batch["wrong_images"].to(device)
            emb = batch["right_embed"].to(device)
            bs = real.size(0)
            ones = torch.ones(bs, device=device)
            zeros = torch.zeros(bs, device=device)

            # ---------------- Discriminator ----------------
            opt_D.zero_grad()
            noise = torch.randn(bs, noise_dim, 1, 1, device=device)
            fake = G(noise, emb)

            real_p, _ = D(real, emb)
            wrong_p, _ = D(wrong, emb)
            fake_p, _ = D(fake.detach(), emb)

            d_loss = bce(real_p, ones) + bce(wrong_p, zeros) + bce(fake_p, zeros)
            d_loss.backward()
            opt_D.step()

            # ---------------- Generator ----------------
            opt_G.zero_grad()
            noise = torch.randn(bs, noise_dim, 1, 1, device=device)
            fake = G(noise, emb)
            fake_p, fake_feat = D(fake, emb)
            _, real_feat = D(real, emb)

            g_bce = bce(fake_p, ones)
            g_l1 = cfg["l1_coeff"] * l1(fake, real)
            g_l2 = cfg["l2_coeff"] * l2(fake_feat.mean(0), real_feat.mean(0).detach())
            g_loss = g_bce + g_l1 + g_l2
            g_loss.backward()
            opt_G.step()

            d_running += d_loss.item()
            g_running += g_loss.item()

        n = len(loader)
        print(f"epoch {epoch}/{cfg['epochs']}  "
              f"D {d_running / n:.3f}  G {g_running / n:.3f}  "
              f"({time.time() - t0:.1f}s)")

        if epoch % cfg["sample_every"] == 0 or epoch == 1:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise, fixed_embed)
            save_sample_grid(samples, f"outputs/samples/epoch_{epoch:04d}.png")
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
