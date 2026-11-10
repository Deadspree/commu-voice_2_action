"""Bidirectional Transformer regressor with explicit time positional encoding."""
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def interpolate_features(features, source_times, target_times):
    if len(source_times) == 1:
        return features[:1].expand(len(target_times), -1)
    upper = torch.searchsorted(source_times.contiguous(), target_times.contiguous()).clamp(1, len(source_times) - 1)
    lower = upper - 1
    weight = ((target_times - source_times[lower]) /
              (source_times[upper] - source_times[lower])).clamp(0, 1)
    return features[lower] + weight[:, None] * (features[upper] - features[lower])


class MotionDecoder(nn.Module):
    def __init__(self, input_width, width=512, layers=4, heads=8, feedforward=2048, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(input_width, width)
        self.register_buffer("frequencies", torch.exp(torch.arange(0, width, 2).float() *
                                                      (-math.log(10000.0) / width)))
        block = nn.TransformerEncoderLayer(width, heads, feedforward, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        # TransformerEncoder clones a layer; initialize each clone independently.
        for layer in self.temporal.layers:
            for name, parameter in layer.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 14))

    def forward(self, features, feature_times, target_times, mask):
        aligned = [interpolate_features(f, t, target) for f, t, target in
                   zip(features, feature_times, target_times)]
        hidden = self.projection(pad_sequence(aligned, batch_first=True))
        times = pad_sequence(target_times, batch_first=True)
        phase = times[:, :, None] * 50 * self.frequencies
        position = torch.zeros_like(hidden)
        position[:, :, 0::2] = phase.sin()
        position[:, :, 1::2] = phase[:, :, :position[:, :, 1::2].shape[-1]].cos()
        hidden = self.temporal(hidden + position, src_key_padding_mask=~mask)
        return self.output(hidden)
