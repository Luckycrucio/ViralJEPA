"""Shared model and data code for exp04: the exp01 CNN-Transformer encoder with a CLS token inside
the Transformer (as in Viraminer3), the JEPA-DNA style token predictor, the BYOL-style CLS heads,
and a GPU batch pipeline over 2-bit base codes (A=0, C=1, G=2, T=3, N=4)."""

import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


WINDOW = 300
CODE = np.full(256, 4, dtype=np.uint8)
for index, base in enumerate(b"ACGT"):
    CODE[base] = index
    CODE[base + 32] = index


# ------------------------------------------------------------------------- data


def read_csv(path):
    """(ids, uint8 codes of shape (n, 300), int labels) from a ViraMiner-format CSV."""
    ids, sequences, labels = [], [], []
    with open(path) as handle:
        for row in csv.reader(handle):
            if len(row) == 3:
                ids.append(row[0])
                sequences.append(row[1][:WINDOW].ljust(WINDOW, "N"))
                labels.append(int(row[2]))
    raw = np.frombuffer("".join(sequences).encode(), dtype=np.uint8).reshape(len(sequences), WINDOW)
    return np.array(ids), CODE[raw], np.array(labels)


def load_corpus(corpus_dir, cache):
    """All pretraining windows as one uint8 array, plus an int64 group per window: windows cut from
    the same record (genome, contig or chromosome) share a group. Both are cached as .npy after
    the first parse; an exp03 codes cache is reused and only the groups are added."""
    cache = Path(cache)
    groups_cache = cache.with_name(cache.stem + "_groups.npy")
    if cache.exists() and groups_cache.exists():
        return np.load(cache), np.load(groups_cache)
    shards = sorted(Path(corpus_dir).glob("pretrain_corpus_*.csv"))
    parsed = [read_csv(shard) for shard in shards]
    codes = np.concatenate([codes for _, codes, _ in parsed])
    # Window id = category|source|record|start|strand; the group key drops start and strand.
    _, groups = np.unique([i.rsplit("|", 2)[0] for ids, _, _ in parsed for i in ids], return_inverse=True)
    if not cache.exists():
        np.save(cache, codes)
    np.save(groups_cache, groups.astype(np.int64))
    return codes, groups.astype(np.int64)


def reverse_complement(codes):
    flipped = codes.flip(-1)
    return torch.where(flipped < 4, 3 - flipped, flipped)


def augment(codes):
    """exp01 augmentation on a GPU batch: reverse complement with p=0.5, then a shift in [-5, 5]
    with N padding."""
    flip = torch.rand(len(codes), device=codes.device) < 0.5
    codes = torch.where(flip[:, None], reverse_complement(codes), codes)
    shift = torch.randint(-5, 6, (len(codes), 1), device=codes.device)
    source = torch.arange(WINDOW, device=codes.device)[None] + shift
    valid = (source >= 0) & (source < WINDOW)
    shifted = codes.gather(1, source.clamp(0, WINDOW - 1))
    return torch.where(valid, shifted, torch.full_like(shifted, 4))


def one_hot(codes):
    return F.one_hot(codes.long(), 5)[..., :4].float()


def batches(codes, labels, batch_size, device, shuffle=False, generator=None):
    order = torch.randperm(len(codes), generator=generator) if shuffle else torch.arange(len(codes))
    for start in range(0, len(codes), batch_size):
        index = order[start : start + batch_size]
        yield codes[index].to(device, non_blocking=True), labels[index].to(device, non_blocking=True)


# ------------------------------------------------------------------------ model


class MultiKernelCNNStem(nn.Module):
    def __init__(self, in_channels=4, filters_per_kernel=72, kernels=(5, 7, 11, 15), stride=2, dropout=0.25):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels, filters_per_kernel, k, stride=stride, padding=k // 2) for k in kernels]
        )
        # LayerNorm, not BatchNorm: no running statistics to drift between masked context batches
        # and the clean EMA target encoder.
        self.norm = nn.LayerNorm(filters_per_kernel * len(kernels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = [F.gelu(conv(x)) for conv in self.convs]
        min_length = min(feature.shape[-1] for feature in features)
        x = torch.cat([feature[..., :min_length] for feature in features], dim=1).transpose(1, 2)
        return self.dropout(self.norm(x))


class ViralJEPAEncoder(nn.Module):
    """exp01 CNN-Transformer encoder with a CLS token prepended before the Transformer, as in
    Viraminer3. Tokens in `token_mask` (CLS excluded, never masked) are replaced by a learnable mask
    token after the stem. The caller must also set their bases to N before the stem, or the
    overlapping kernels of neighbouring tokens still see them."""

    def __init__(self, cnn_dropout=0.25, transformer_dropout=0.1, transformer_dim=288, filters_per_kernel=72,
                 kernels=(5, 7, 11, 15), cnn_stride=2, num_layers=6, nhead=8, max_len=WINDOW):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.stem = MultiKernelCNNStem(4, filters_per_kernel, kernels, cnn_stride, cnn_dropout)
        stem_dim = filters_per_kernel * len(kernels)
        self.projection = nn.Identity() if stem_dim == transformer_dim else nn.Linear(stem_dim, transformer_dim)
        self.token_count = min(((max_len + 2 * (k // 2) - k) // cnn_stride) + 1 for k in kernels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)
        self.pos_enc = nn.Parameter(torch.randn(1, self.token_count + 1, transformer_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(transformer_dim, nhead, transformer_dim * 4, transformer_dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(transformer_dim)

    def forward(self, x, token_mask=None, return_blocks=0):
        """(cls, tokens) after the final LayerNorm. With `return_blocks` = K, also the raw token
        outputs (CLS dropped) of the last K Transformer blocks, for data2vec-style targets."""
        x = self.projection(self.stem(x))
        if token_mask is not None:
            x = torch.where(token_mask[..., None], self.mask_token.to(x.dtype), x)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1).to(x.dtype), x], dim=1) + self.pos_enc
        blocks = []
        for layer in self.transformer.layers:
            x = layer(x)
            blocks.append(x[:, 1:])
        x = self.norm(x)
        if return_blocks:
            return x[:, 0], x[:, 1:], blocks[-return_blocks:]
        return x[:, 0], x[:, 1:]


def sinusoidal(count, dim):
    position = torch.arange(count)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    table = torch.zeros(count, dim)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table[None]


class JEPAPredictor(nn.Module):
    """JEPA-DNA predictor: 4 pre-norm layers at half the encoder width, 3 heads, frozen sinusoidal
    positions. Takes the context encoder's CLS and tokens; masked tokens are replaced by a learnable
    mask token, and every token's output is projected back to the encoder width (I-JEPA style
    per-token prediction)."""

    def __init__(self, dim=288, token_count=150, layers=4, heads=3, dropout=0.0):
        super().__init__()
        width = dim // 2
        self.embed = nn.Linear(dim, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.register_buffer("pos", sinusoidal(token_count + 1, width), persistent=False)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, cls, tokens, token_mask):
        """Predicted representation of every token (the caller scores the masked ones)."""
        x = self.embed(torch.cat([cls[:, None], tokens], dim=1))
        mask = F.pad(token_mask, (1, 0))  # CLS is never masked
        x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        x = self.norm(self.transformer(x + self.pos))
        return self.out(x[:, 1:])


def mlp(*dims):
    """BYOL projector/predictor: Linear-LayerNorm-GELU blocks and a plain Linear output, so the
    output scale is free (LayerNorm in place of BYOL's BatchNorm, which would drift between the
    online and EMA copies)."""
    layers = []
    for width_in, width_out in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(width_in, width_out), nn.LayerNorm(width_out), nn.GELU()]
    return nn.Sequential(*layers, nn.Linear(dims[-2], dims[-1]))


class ViralJEPAClassifier(nn.Module):
    """Viraminer3 head: the encoder's CLS output (already LayerNorm'd), then an MLP."""

    def __init__(self, encoder, dropout=0.25, transformer_dim=288):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(transformer_dim, 512), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(512),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.fc(self.encoder(x)[0])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def autocast(device):
    """bf16 on Ampere or newer; fp32 elsewhere (pre-Ampere GPUs only emulate bf16)."""
    enabled = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)
