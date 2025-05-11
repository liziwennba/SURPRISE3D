export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME=/workspace/huggingface
source /opt/conda/bin/activate /opt/conda/envs/Intent3D
sh init.sh
sh scripts/train_scanintend.sh