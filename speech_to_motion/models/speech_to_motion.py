from torch import nn

from .hubert_encoder import HubertEncoder
from .motion_decoder import MotionDecoder


class SpeechToMotion(nn.Module):
    def __init__(self, cfg, hubert_config=None):
        super().__init__()
        self.encoder = HubertEncoder(cfg["encoder"]["name"], cfg["encoder"]["revision"], hubert_config)
        self.decoder = MotionDecoder(self.encoder.width, **cfg["decoder"])

    def forward(self, batch):
        if "features" in batch:
            if self.encoder.unfreeze_last_n:
                raise ValueError("Fine-tuning cannot use cached features")
            features, times = batch["features"], batch["feature_times"]
        else:
            encoded = self.encoder(batch["audio"])
            features, times = zip(*encoded)
        return self.decoder(features, times, batch["times"], batch["mask"])
