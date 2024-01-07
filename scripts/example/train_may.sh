SEQ=may

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
   --nproc_per_node=4 --master_port 2132 \
   train.py configs/face_simple_configs/may/${SEQ}.yaml
# CUDA_VISIBLE_DEVICES=0 python3 train.py configs/face_simple_configs/may/${SEQ}.yaml