"""
DistilBERT text encoder.

We use a *frozen* DistilBERT to turn a flower caption (e.g. "this flower has
soft pink petals with a yellow center") into a single 768-dimensional vector.
That vector is the "text condition" the GAN-CLS generator and discriminator
both consume.

Why frozen? Training BERT *and* a GAN at the same time on a small dataset is
unstable and slow. The standard recipe (and the one in your reference repo) is
to precompute the embeddings once and treat them as fixed features.

Why mean-pooling? DistilBERT has no [CLS] "pooler" head like full BERT, so the
common, robust choice is to average the token vectors (respecting the
attention mask so padding tokens don't count).
"""

import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

# DistilBERT hidden size. This is the `embed_dim` the GAN consumes.
EMBED_DIM = 768


class TextEncoder:
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cpu"):
        self.device = device
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(device).eval()
        # Freeze: we never backprop into BERT.
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts, max_length: int = 64) -> torch.Tensor:
        """texts: list[str] -> tensor of shape (len(texts), 768) on CPU."""
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        last_hidden = self.model(**enc).last_hidden_state          # (B, T, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()          # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)                    # (B, 768)
        counts = mask.sum(dim=1).clamp(min=1e-9)                    # (B, 1)
        return (summed / counts).cpu()                             # mean-pooled
