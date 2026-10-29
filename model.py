"""Shared model and data code for exp02: the exp01 CNN-Transformer encoder, the JEPA-DNA style
predictor, and a GPU batch pipeline over 2-bit base codes (A=0, C=1, G=2, T=3, N=4)."""

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
    """All pretraining windows as one uint8 array, cached as .npy after the first parse."""
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    shards = sorted(Path(corpus_dir).glob("pretrain_corpus_*.csv"))
    codes = np.concatenate([read_csv(shard)[1] for shard in shards])
    np.save(cache, codes)
    return codes


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
        self.norm = nn.BatchNorm1d(filters_per_kernel * len(kernels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = [F.gelu(conv(x)) for conv in self.convs]
        min_length = min(feature.shape[-1] for feature in features)
        x = torch.cat([feature[..., :min_length] for feature in features], dim=1)
        return self.dropout(self.norm(x)).transpose(1, 2)


class ViralJEPAEncoder(nn.Module):
    """exp01 CNN-Transformer encoder (unchanged), returning token-level representations."""

    def __init__(self, cnn_dropout=0.25, transformer_dropout=0.1, transformer_dim=288, filters_per_kernel=72,
                 kernels=(5, 7, 11, 15), cnn_stride=2, num_layers=6, nhead=8, max_len=WINDOW):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.stem = MultiKernelCNNStem(4, filters_per_kernel, kernels, cnn_stride, cnn_dropout)
        stem_dim = filters_per_kernel * len(kernels)
        self.projection = nn.Identity() if stem_dim == transformer_dim else nn.Linear(stem_dim, transformer_dim)
        self.token_count = min(((max_len + 2 * (k // 2) - k) // cnn_stride) + 1 for k in kernels)
        self.pos_enc = nn.Parameter(torch.randn(1, self.token_count, transformer_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(transformer_dim, nhead, transformer_dim * 4, transformer_dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(transformer_dim)

    def forward_tokens(self, x):
        x = self.projection(self.stem(x))
        x = self.transformer(x + self.pos_enc[:, : x.shape[1]])
        return self.norm(x)

    def forward(self, x):
        return self.forward_tokens(x)


def sinusoidal(count, dim):
    position = torch.arange(count)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    table = torch.zeros(count, dim)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table[None]


class JEPAPredictor(nn.Module):
    """JEPA-DNA predictor: 4 pre-norm layers at half the encoder width, 3 heads, frozen sinusoidal
    positions. Masked token outputs are replaced by a learnable mask token; the mean-pooled output
    is projected back to the encoder width as the predicted sequence representation."""

    def __init__(self, dim=288, token_count=150, layers=4, heads=3, dropout=0.0):
        super().__init__()
        width = dim // 2
        self.embed = nn.Linear(dim, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.register_buffer("pos", sinusoidal(token_count, width), persistent=False)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, tokens, token_mask=None):
        x = self.embed(tokens)
        if token_mask is not None:
            x = torch.where(token_mask[..., None], self.mask_token.to(x.dtype), x)
        x = self.norm(self.transformer(x + self.pos))
        return self.out(x.mean(1))


class ViralJEPAClassifier(nn.Module):
    """exp01 classification head: one CLS attention layer over the encoder tokens, then an MLP."""

    def __init__(self, encoder, dropout=0.25, transformer_dim=288):
        super().__init__()
        self.encoder = encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)
        self.cls_pos = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)
        self.cls_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(transformer_dim, 8, transformer_dim * 4, 0.1, activation="gelu",
                                       batch_first=True, norm_first=True),
            num_layers=1,
        )
        self.norm = nn.LayerNorm(transformer_dim)
        self.fc = nn.Sequential(
            nn.Linear(transformer_dim, 512), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(512),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x):
        tokens = self.encoder.forward_tokens(x)
        cls = self.cls_token.expand(len(tokens), -1, -1) + self.cls_pos
        tokens = self.cls_attention(torch.cat([cls, tokens], dim=1))
        return self.fc(self.norm(tokens[:, 0]))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def autocast(device):
    """bf16 on Ampere or newer; fp32 elsewhere (pre-Ampere GPUs only emulate bf16)."""
    enabled = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)
