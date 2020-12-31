CUDA_VISIBLE_DEVICES=0,1
# export PYTHONPATH=/home/anhdh/projects/pysot2:$PYTHONPATH
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml