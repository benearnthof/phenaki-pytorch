batch = next(iter(trainer.ds))
batch
print(f"Tensor min: {batch.min():.3f}, max: {batch.max():.3f}, mean: {batch.mean():.3f}")

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="obvious-research/phenaki-cvivit",
    allow_patterns="frozen_models/*",
    local_dir="/workspace/phenaki-cvivit/frozen_models/"
)

import argparse
from phenaki_pytorch import CViViT, CViViTTrainer
import os

pip install lpips