CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py \
    --datasets rdvs --data_root /home/linj/workspace/vsod/datasets

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py \
    --datasets vidsod_100 --data_root /home/linj/workspace/vsod/datasets

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py \
    --datasets dvisal --data_root /home/linj/workspace/vsod/datasets
