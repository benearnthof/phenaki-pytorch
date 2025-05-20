git clone https://github.com/benearnthof/phenaki-pytorch.git

# CELEBV-HQ
cd /workspace/datasets
aria2c "magnet:?xt=urn:btih:843b5adb0358124d388c4e9836654c246b988ff4&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php%3Fpasskey%3D59191383faf97bc1bf5459852ce2acef&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
cd ./CelebV-HQ

apt-get install pigz

pigz -dc celebvhq.tar.gz | tar -xvf - --no-same-owner

find ./35666 -type f | wc -l

source /workspace/venv/bin/activate

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
pip install google-cloud-storage
pip install nibabel
pip install prettytable
pip install tensorhue
# setup gcloud
export GOOGLE_APPLICATION_CREDENTIALS="/workspace/ldm100k-dl-95af770e42ef.json"

apt-get install apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update && apt-get install google-cloud-cli

gcloud init
gcloud auth login --no-launch-browser
gcloud config set project ldm100k
gcloud storage ls gs://ldm100k-bucket

gsutil cat gs://ldm100k-bucket/shards/shard-0000000.tar

echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt -qq update
apt -qq install fuse gcsfuse

apt-get update && apt-get install -y fuse kmod
apt-get update && apt-get install -y fuse gcsfuse

# mounting virtual drive
cd /workspace/
mkdir -p /workspace/mnt/ldm100k
gcsfuse --implicit-dirs ldm100k-bucket /workspace/mnt/ldm100k

# need explicit list of shards since we cannot mount on vm
gsutil ls gs://ldm100k-bucket/shards/ > ldm100k_shards.txt

gsutil iam ch allUsers:objectViewer gs://ldm100k-bucket

# signurls
# gsutil signurl -d 1h /workspace/ldm100k-dl-95af770e42ef.json gs://ldm100k-bucket/shards/shard-*.tar
