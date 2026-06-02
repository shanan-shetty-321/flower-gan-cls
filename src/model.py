"""
GAN-CLS generator and discriminator (DCGAN backbone), conditioned on a
768-d DistilBERT caption embedding. Output resolution is 64x64 RGB.

Shapes flow like this:
  Generator:  noise(100) + text(768->embed_out) -> 4x4 -> 8 -> 16 -> 32 -> 64
  Discriminator: 64x64 img -> 32 -> 16 -> 8 -> 4 (512 ch), then text is tiled
                 over the 4x4 grid and concatenated, then a final conv -> 1.

The discriminator returns BOTH a real/fake probability AND its 512x4x4 feature
map. The feature map is used for the generator's L2 "feature matching" loss,
which stabilizes training (push fake-image features toward real-image features).
"""

import torch
import torch.nn as nn

from .text_encoder import EMBED_DIM


class Generator(nn.Module):
    def __init__(self, channels: int = 3, noise_dim: int = 100,
                 embed_dim: int = EMBED_DIM, embed_out_dim: int = 128):
        super().__init__()
        self.noise_dim = noise_dim

        # Project the 768-d caption embedding down to a compact conditioning vector.
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_out_dim),
            nn.BatchNorm1d(embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        def block(in_c, out_c, k=4, s=2, p=1, last=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False)]
            if last:
                layers += [nn.Tanh()]                 # output in [-1, 1]
            else:
                layers += [nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            return layers

        self.model = nn.Sequential(
            *block(noise_dim + embed_out_dim, 512, k=4, s=1, p=0),  # -> 4x4
            *block(512, 256),                                       # -> 8x8
            *block(256, 128),                                       # -> 16x16
            *block(128, 64),                                        # -> 32x32
            *block(64, channels, last=True),                        # -> 64x64
        )

    def forward(self, noise, text):
        # noise: (B, noise_dim, 1, 1)   text: (B, embed_dim)
        t = self.text_proj(text)                 # (B, embed_out_dim)
        t = t.view(t.size(0), t.size(1), 1, 1)        # (B, embed_out_dim, 1, 1)
        z = torch.cat([t, noise], dim=1)              # (B, noise+embed_out, 1, 1)
        return self.model(z)                          # (B, 3, 64, 64)


class Discriminator(nn.Module):
    def __init__(self, channels: int = 3,
                 embed_dim: int = EMBED_DIM, embed_out_dim: int = 128):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if normalize:
                layers += [nn.BatchNorm2d(out_c)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers

        # Image feature extractor: 64x64 -> 4x4, 512 channels.
        self.features = nn.Sequential(
            *block(channels, 64, normalize=False),    # -> 32x32
            *block(64, 128),                          # -> 16x16
            *block(128, 256),                         # -> 8x8
            *block(256, 512),                         # -> 4x4
        )

        # Project the caption to embed_out_dim, then tile across the 4x4 grid.
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_out_dim),
            nn.BatchNorm1d(embed_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512 + embed_out_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, image, text):
        feats = self.features(image)                  # (B, 512, 4, 4)
        t = self.text_proj(text)                 # (B, embed_out_dim)
        t = t.view(t.size(0), t.size(1), 1, 1)
        t = t.repeat(1, 1, feats.size(2), feats.size(3))  # tile -> (B, eo, 4, 4)
        x = torch.cat([feats, t], dim=1)              # (B, 512+eo, 4, 4)
        prob = self.classifier(x).view(-1)            # (B,)  real/fake score
        return prob, feats                            # feats used for feat-matching
