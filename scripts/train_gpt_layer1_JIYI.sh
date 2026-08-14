CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29500 ../main_motion_gpt.py --config ../Config/motion_gpt_train_jiyi.yaml --train > gpt_layer_1.txt 2>&1 &

# debug
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 ../main_motion_gpt.py --config ../Config/motion_gpt_train_jiyi.yaml --train 
