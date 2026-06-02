"""
GAN-CLS dataset wrapper around the HuggingFace `pranked03/flowers-blip-captions`
dataset (Oxford-102 flowers + BLIP captions).

The "CLS" (matching-aware) trick needs three things per sample:
  1. right_images : the real image for this caption        -> label "real & matching"
  2. right_embed  : the caption embedding (precomputed)
  3. wrong_images : a real image from a DIFFERENT flower class -> label "mismatched"

Pairing the right caption with a wrong image and training the discriminator to
reject it is what forces the model to learn text-image *alignment*, not just
"is this a realistic flower".

Caption embeddings are precomputed ONCE (DistilBERT is frozen) and passed in.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class Text2ImageDataset(Dataset):
    def __init__(self, hf_split, embeddings: torch.Tensor, transform=None):
        self.ds = hf_split
        self.embeddings = embeddings              # (N, 768) precomputed
        self.transform = transform
        # Labels are cheap ints; cache them so wrong-pair sampling never decodes images.
        self.labels = list(hf_split["label"])
        self.n = len(self.labels)

    def __len__(self):
        return self.n

    def _to_tensor(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)

    def _sample_wrong_index(self, label):
        # Pick a random index whose class differs from `label`.
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
