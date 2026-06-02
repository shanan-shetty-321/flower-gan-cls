"""
Generate flower images from a text prompt using a trained generator.

    python -m src.generate --prompt "a flower with bright yellow petals" --n 8

Works fine on CPU (inference is cheap). Saves a grid to outputs/.
"""

import argparse
import torch

from .model import Generator
from .text_encoder import TextEncoder, EMBED_DIM
from .utils import save_sample_grid


def load_generator(ckpt_path, device, noise_dim=100, embed_out_dim=128):
    G = Generator(channels=3, noise_dim=noise_dim,
                  embed_dim=EMBED_DIM, embed_out_dim=embed_out_dim).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.eval()
    return G


@torch.no_grad()
def generate(prompt, ckpt="models/generator_best.pth", n=8,
             noise_dim=100, embed_out_dim=128, out="outputs/generated.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TextEncoder(device=str(device))
    G = load_generator(ckpt, device, noise_dim, embed_out_dim)

    emb = encoder.encode([prompt]).to(device).repeat(n, 1)   # same caption, n noises
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
