conda activate tensorml

git config --global credential.helper store
sudo apt install git-lfs
git lfs install

git clone https://github.com/photomz/outlier-free-transformers.git

echo 'alias skyy="cd ~/outlier-free-transformers && conda activate tensorml"' >> ~/.bashrc
source ~/.bashrc

skyy
git checkout tensordock

# README install
skyy
pip install --upgrade --no-deps pip
pip install torch==1.13.0 torchvision==0.16.0
pip install -r docker/requirements.txt
wandb login
chmod +x scripts/*.sh