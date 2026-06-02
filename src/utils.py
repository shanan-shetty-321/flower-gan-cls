"""Shared helpers: image transforms, weight init, checkpoint + sample saving."""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils

IMAGE_SIZE = 64

# Images -> 64x64 tensors normalized to [-1, 1] (matches the generator's Tanh).
transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),                                  # [0,1]
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # -> [-1,1]
])


def weights_init(m):
    """DCGAN init: conv weights ~N(0,0.02), BN weights ~N(1,0.02)."""
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
    """[-1,1] -> [0,1] for viewing/saving."""
    return (t.clamp(-1, 1) + 1) / 2


def save_sample_grid(images, path, nrow=4):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    vutils.save_image(denorm(images), path, nrow=nrow)
