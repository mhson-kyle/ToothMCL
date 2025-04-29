now=$(date +"%Y%m%d_%H%M%S")
name=ours
pretrained_root=pretrained
logdir=runs/logs_swin_base_ours
feature_size=48
data_dir=./Dataset116_ToothFairy2cropped
cache_dir=./Dataset116_ToothFairy2cropped/cache
use_ssl_pretrained=True
use_persistent_dataset=True

mkdir -p $logdir

CUDA_VISIBLE_DEVICES=0 python main.py \
    --name $name \
    --distributed \
    --pretrained_root $pretrained_root \
    --feature_size $feature_size \
    --data_dir $data_dir \
    --cache_dir $cache_dir \
    --use_ssl_pretrained $use_ssl_pretrained \
    --use_persistent_dataset $use_persistent_dataset \
    --logdir $logdir | tee $logdir/$now.txt