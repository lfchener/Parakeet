
# train model
export CUDA_VISIBLE_DEVICES=7
#nvprof -o tacotron2_timeline -f --cpu-profiling off --profile-from-start off \
python -u train.py \
--use_gpu=1 \
--data='../../dataset/LJSpeech-1.1' \
--output='./experiment' \
--config='configs/ljspeech.yaml' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
