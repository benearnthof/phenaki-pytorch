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


# ldm100k webdataset
import os

# This sets the environment variable for the entire Python process (and subprocesses like gsutil, gcsfuse)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/ldm100k-dl-95af770e42ef.json"

import webdataset as wds

with open("/workspace/ldm100k_shards.txt") as f:
    urls = [line.strip().replace("gs://ldm100k-bucket", "https://storage.googleapis.com/ldm100k-bucket") for line in f]

urls = [x for x in urls if x.endswith(".tar")]

dataset = wds.WebDataset(urls, shardshuffle=False)

import torch
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
sample = next(iter(dataloader))

dl_iter = iter(dataloader)

import json

for i in range(0, 100, 1):
    sample = next(dl_iter)
    metadata = [json.loads(x) for x in sample["json"]]
    print(f"Sample {i}: {metadata[0].get('age')}")

sample["json"]
sample["__key__"]

import glob
import nibabel as nib

import io
import gzip
from tensorhue import viz

img_bytes = sample["image.nii.gz"][0]

with gzip.open(io.BytesIO(img_bytes), "rb") as gz:
    decompressed = gz.read()

def save_nii_gz(bytes_data: bytes, output_path: str):
    with open(output_path, 'wb') as f:
        f.write(bytes_data)

# Example usage
save_nii_gz(decompressed, 'output_file.nii.gz')

xd = nib.load("output_file.nii.gz")

from nibabel import FileHolder, Nifti1Image
# for gzipped files
from gzip import GzipFile
fh = FileHolder(fileobj=GzipFile(fileobj=io.BytesIO(decompressed)))
img = Nifti1Image.from_file_map({"header": fh, "image": fh})

dta = img.get_fdata()
dta.shape

img_slice = dta[:, 110, :]
img_slice.shape

from scipy.ndimage import zoom
# For 2D array, zoom factor 0.5 reduces size by half
subsampled = zoom(img_slice, zoom=0.5, order=1)  # order=1 is bilinear interpolation

viz(subsampled)


# In memory webdataset

import os
import webdataset as wds

# Path to your raw video files
video_folder = "/workspace/datasets/CelebV-HQ/35666"

# Collect all mp4 paths
video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")]

# Create a synthetic webdataset-style iterable
def generator():
    for path in video_paths:
        key = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as f:
            video_bytes = f.read()
        # Emit a "sample" in WebDataset format
        yield {
            "__key__": key,
            "mp4": video_bytes
        }

# Wrap with webdataset
dataset = wds.DataPipeline(
    generator,
    wds.to_tuple("mp4"),  # you can adjust this depending on your model’s expected tuple format
)
