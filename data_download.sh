mkdir train
cd train
wget https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
wget https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
wget https://www.statmt.org/wmt13/training-parallel-un.tgz
wget https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
wget https://www.statmt.org/wmt10/training-giga-fren.tar
gunzip *.tgz
for f in *.tar; do echo "Extracting $f"; tar -xf "$f"; done
gunzip *.gz