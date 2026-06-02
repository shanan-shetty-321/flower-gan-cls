"""
Local CPU sanity check — NOT real training. Confirms the whole pipeline wires
together (DistilBERT -> embeddings -> dataset -> G/D -> losses -> backprop) on a
tiny subset in a few seconds, before you waste GPU hours on Kaggle.

    python smoke_test.py
"""

import torch
import torch.nn as nn
from datasets import load_dataset

from src.dataset import Text2ImageDataset
from src.model import Generator, Discriminator
from src.text_encoder import EMBED_DIM
from src.utils import transform, weights_init

N = 2           # minimum that works with BatchNorm (needs batch > 1)
BATCH = 2
EMBED_OUT = 128
NOISE = 100


def main():
    device = torch.device("cpu")
    print("Loading a tiny slice of the dataset...")
    ds = load_dataset("pranked03/flowers-blip-captions", split=f"train[:{N}]")

    # Use random embeddings instead of DistilBERT — proves shapes/losses are correct
    # without waiting for a slow CPU transformer forward pass.
    emb = torch.randn(N, EMBED_DIM)
    print(f"  embeddings OK (random, skipping DistilBERT for speed): {tuple(emb.shape)}")

    dataset = Text2ImageDataset(ds, emb, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=True)
    batch = next(iter(loader))
    assert batch["right_images"].shape == (BATCH, 3, 64, 64), batch["right_images"].shape
    print(f"  batch images OK: {tuple(batch['right_images'].shape)}")

    G = Generator(3, NOISE, EMBED_DIM, EMBED_OUT); G.apply(weights_init)
    D = Discriminator(3, EMBED_DIM, EMBED_OUT); D.apply(weights_init)
    bce, l1, l2 = nn.BCELoss(), nn.L1Loss(), nn.MSELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    print("Running 2 training steps...")
    for step in range(2):
        real = batch["right_images"]; wrong = batch["wrong_images"]; e = batch["right_embed"]
        bs = real.size(0)
        ones, zeros = torch.ones(bs), torch.zeros(bs)

        opt_D.zero_grad()
        fake = G(torch.randn(bs, NOISE, 1, 1), e)
        rp, _ = D(real, e); wp, _ = D(wrong, e); fp, _ = D(fake.detach(), e)
        d_loss = bce(rp, ones) + bce(wp, zeros) + bce(fp, zeros)
        d_loss.backward(); opt_D.step()

        opt_G.zero_grad()
        fake = G(torch.randn(bs, NOISE, 1, 1), e)
        fp, ff = D(fake, e); _, rf = D(real, e)
        g_loss = bce(fp, ones) + 50 * l1(fake, real) + 100 * l2(ff.mean(0), rf.mean(0).detach())
        g_loss.backward(); opt_G.step()
        print(f"  step {step}: D={d_loss.item():.3f}  G={g_loss.item():.3f}")

    print("\nSMOKE TEST PASSED — pipeline is correct. Ready to train on Kaggle.")


if __name__ == "__main__":
    main()
