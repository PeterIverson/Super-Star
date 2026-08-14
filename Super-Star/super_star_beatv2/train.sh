# audio_casual
# emage vq
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m train --cfg configs/config_mixed_stage3_a2m_audio_causal.yaml --nodebug

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m train --cfg configs/config_mixed_stage3_a2m_audio_causal.yaml --nodebug > gpt_lom_post_train_audio_causal.txt 2>&1 &
