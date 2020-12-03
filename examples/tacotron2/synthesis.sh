export CUDA_VISIBLE_DEVICES=0
python -u synthesis.py \
--use_gpu=1 \
--output='./synthesis' \
--config='configs/ljspeech.yaml' \
--checkpoint='./experiment_log/checkpoints/step-22000' \
--vocoder='waveflow' \
--config_vocoder='./waveflow_res128_ljspeech_ckpt_1.0/waveflow_ljspeech.yaml' \
--checkpoint_vocoder='./waveflow_res128_ljspeech_ckpt_1.0/step-2000000' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
