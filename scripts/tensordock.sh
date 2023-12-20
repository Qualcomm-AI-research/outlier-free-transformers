if [ $? -ne 0 ]; then
	conda create -n agi python=3.8 -y
	conda activate agi
fi

git config --global credential.helper store
sudo apt install git-lfs
git lfs install

git clone https://github.com/photomz/outlier-free-transformers.git
git checkout tensordock

echo 'alias skyy="cd ~/outlier-free-transformers && conda activate agi"' >> ~/.bashrc
source ~/.bashrc

# README install
cd ~/sky_workdir
pip install --upgrade --no-deps pip
pip install torch==1.11.0 torchvision==0.12.0
pip install -r docker/requirements.txt
wandb login
chmod +x scripts/*.sh