export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME=/workspace/huggingface
source /opt/conda/bin/activate /opt/conda/envs/spatial
python -m torch.distributed.run --nproc_per_node=8 --master_port=29501 train.py --cfg-path lavis/projects/reason3d/train/reason3d_scanrefer_scratch.yaml