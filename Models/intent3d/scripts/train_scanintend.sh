export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME=/workspace/huggingface
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node 8 --master_port 3123 \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root /root/scannet++ \
    --val_freq 1 --batch_size 6 --print_freq 10 \
    --lr_backbone=1e-3 --lr=1e-4 --text_encoder_lr=1e-5 --lr_decay_epochs 65 \
    --dataset scannetpp --test_dataset scannetpp \
    --detect_intermediate \
    --use_soft_token_loss --use_contrastive_align  \
    --log_dir /root/spatial/Intent3D/data/output/IntentNet/wp \
    --tensorboard /root/spatial/Intent3D/data/tensorboard/IntentNet/wp \
    --pp_checkpoint /root/spatial/Intent3D/gf_detector_l6o256.pth \
    --butd --self_attend --augment_det \
    --max_epoch 90 --eval_batch_size 8 --num_workers 8 --num_target 50 \
    # --checkpoint_path /data/kangweitai/3DIG/output/IntentNet/scanintend/1731989849/ckpt_last.pth