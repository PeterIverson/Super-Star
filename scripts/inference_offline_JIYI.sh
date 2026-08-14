#!/bin/bash
# specify the folder path to iterate over
AUDIO_DIR="../jiyi_test/"

for filepath in "${AUDIO_DIR}"/*.wav
do
    # extract the file name (without path and extension)
    file=$(basename "${filepath}" .wav)

    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29502 generate_gestures.py \
      --audio_path "${AUDIO_DIR}/${file}.wav" \
      --save_dir '../results_offline' \
      --eval_dir '../data_test/pred_motion_offline' \
      --rqvae_path '../pretrained_models/rqvae.pt' \
      --model_path_0 '../pretrained_models/base_0.pt' \
      --model_path_1 '../pretrained_models/finetuning_1.pt' \
      --model_path_2 '../pretrained_models/finetuning_2.pt' \
      --model_path_3 '../pretrained_models/finetuning_3.pt' \
      --init_body_pose_code 492 \
      --init_hands_pose_code 423 \
      --processed_dataset_dir '../Data/JIYI_processed'\
done
