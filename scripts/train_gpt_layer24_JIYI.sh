CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29501 ../main_motion_gpt.py --config ../Config/motion_fine_gpt_rq_level_1_train_jiyi.yaml --train > gpt_layer_2_cross.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29502 ../main_motion_gpt.py --config ../Config/motion_fine_gpt_rq_level_2_train_jiyi.yaml --train > gpt_layer_3_cross.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29503 ../main_motion_gpt.py --config ../Config/motion_fine_gpt_rq_level_3_train_jiyi.yaml --train > gpt_layer_4_cross.txt 2>&1 &
