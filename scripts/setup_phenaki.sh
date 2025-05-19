git clone https://github.com/benearnthof/phenaki-pytorch.git

# CELEBV-HQ
cd /workspace/datasets
aria2c "magnet:?xt=urn:btih:843b5adb0358124d388c4e9836654c246b988ff4&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php%3Fpasskey%3D59191383faf97bc1bf5459852ce2acef&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
cd ./CelebV-HQ

apt-get install pigz

pigz -dc celebvhq.tar.gz | tar -xvf - --no-same-owner

find ./35666 -type f | wc -l

cd /workspace/phenaki-pytorch
pip install -e .

pip install accelerate
pip install beartype
pip install einops>=0.6
pip install ema-pytorch>=0.1.1
pip install opencv-python
pip install pillow
pip install numpy
pip install plotly
pip install webdataset
pip install sentencepiece
pip install torch==1.13.1
pip install torchtyping
pip install torchvision
pip install transformers>=4.20.1
pip install tqdm
pip install vector-quantize-pytorch>=0.10.15
pip install wandb
pip install av
pip install lpips