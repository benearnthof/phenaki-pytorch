git clone https://github.com/benearnthof/phenaki-pytorch.git

# CELEBV-HQ
cd /workspace/datasets
aria2c "magnet:?xt=urn:btih:843b5adb0358124d388c4e9836654c246b988ff4&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php%3Fpasskey%3D59191383faf97bc1bf5459852ce2acef&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
cd ./CelebV-HQ

apt-get install pigz

pigz -dc celebvhq.tar.gz | tar -xvf - --no-same-owner

find ./35666 -type f | wc -l
