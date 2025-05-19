# example script to test training on CelebV-HQ

import torch
from phenaki_pytorch import CViViT, CViViTTrainer

cvivit = CViViT(
    dim = 512,
    codebook_size = 65536,
    image_size = 128,
    patch_size = 16,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
).cuda()

trainer = CViViTTrainer(
    cvivit,
    folder = '/workspace/datasets/CelebV-HQ/35666',
    batch_size = 4,
    grad_accum_every = 4,
    train_on_images = False,  # you can train on images first, before fine tuning on video, for sample efficiency
    use_ema = False,          # recommended to be turned on (keeps exponential moving averaged cvivit) unless if you don't have enough resources
    num_train_steps = 10000
)

trainer.train()               # reconstructions and checkpoints will be saved periodically to ./results

