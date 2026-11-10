"""HuBERT wrapper with explicit freeze policy and convolutional time coordinates."""
from collections import defaultdict

import torch
from torch import nn


class HubertEncoder(nn.Module):
    def __init__(self, name=None, revision=None, hubert_config=None):
        super().__init__()
        from transformers import HubertConfig, HubertModel
        if hubert_config is not None:
            self.model = HubertModel(HubertConfig.from_dict(hubert_config))
        else:
            self.model = HubertModel.from_pretrained(name, revision=revision)
        self.model.config.apply_spec_augment = False
        self.model.config.layerdrop = 0.0
        self.unfreeze_last_n = 0
        self.set_trainable(0)

    @property
    def width(self):
        return self.model.config.hidden_size

    def set_trainable(self, last_n, checkpointing=False):
        layers = self.model.encoder.layers
        if not 0 <= last_n <= len(layers):
            raise ValueError(f"unfreeze_last_n must be between 0 and {len(layers)}")
        self.unfreeze_last_n = last_n
        self.model.requires_grad_(False)
        if last_n:
            for layer in layers[-last_n:]:
                layer.requires_grad_(True)
            # Stable-layer-norm HuBERT variants apply this after all layers.
            if self.model.config.do_stable_layer_norm:
                self.model.encoder.layer_norm.requires_grad_(True)
        if last_n and checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.model.gradient_checkpointing_disable()
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen layers deterministic even while the decoder is training.
        self.model.eval()
        if mode and self.unfreeze_last_n:
            self.model.train()
            for module in (self.model.feature_extractor, self.model.feature_projection,
                           self.model.encoder.pos_conv_embed, self.model.encoder.layer_norm,
                           self.model.encoder.dropout):
                module.eval()
            for layer in self.model.encoder.layers[:-self.unfreeze_last_n]:
                layer.eval()
        return self

    def feature_times(self, length, device):
        jump, receptive = 1, 1
        for kernel, stride in zip(self.model.config.conv_kernel, self.model.config.conv_stride):
            receptive += (kernel - 1) * jump
            jump *= stride
        return (torch.arange(length, device=device, dtype=torch.float32) * jump +
                (receptive - 1) / 2) / 16000

    def forward(self, waveforms):
        # Group exact lengths: no artificial silence/padding can contaminate
        # group-normalized HuBERT features, and cached/online inputs agree.
        groups = defaultdict(list)
        for i, waveform in enumerate(waveforms):
            groups[len(waveform)].append(i)
        output = [None] * len(waveforms)
        for length, indices in groups.items():
            if length < 400:
                raise ValueError("HuBERT audio windows need at least 400 samples")
            audio = torch.stack([waveforms[i] for i in indices]).float()
            audio = (audio - audio.mean(-1, keepdim=True)) / torch.sqrt(
                audio.var(-1, unbiased=False, keepdim=True) + 1e-7)
            with torch.set_grad_enabled(self.training and self.unfreeze_last_n > 0):
                features = self.model(audio).last_hidden_state
            times = self.feature_times(features.shape[1], features.device)
            for offset, index in enumerate(indices):
                output[index] = (features[offset], times)
        return output
