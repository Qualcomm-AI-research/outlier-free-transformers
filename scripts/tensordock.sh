conda create -n agi python=3.9 -y

git config --global credential.helper store
sudo apt install git-lfs
git lfs install

git clone https://github.com/photomz/outlier-free-transformers.git

echo 'alias skyy="cd ~/outlier-free-transformers && conda activate agi"' >> ~/.bashrc
echo 'export OMP_NUM_THREADS=$(($(nproc) - 2))' >> ~/.bashrc # num CPU cores-1
# echo 'export LD_LIBRARY_PATH=/home/user/mambaforge/condabin/conda:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

skyy
git checkout tensordock

# README install
skyy
pip install -r docker/requirements.txt
wandb login