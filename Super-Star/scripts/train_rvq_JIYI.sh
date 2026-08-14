CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29500 ../main_motion_vqvae.py --config ../Config/motion_rvq_train_jiyi.yaml --train > rqvae_log.txt 2>&1 &
