export CUDA_VISIBLE_DEVICES=0
python -u synthesis.py \
--use_gpu=1 \
--output='./synthesis' \
--config='configs/ljspeech.yaml' \
--checkpoint='./experiment_log/checkpoints/step-10000' \
--vocoder='griffin-lim' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
