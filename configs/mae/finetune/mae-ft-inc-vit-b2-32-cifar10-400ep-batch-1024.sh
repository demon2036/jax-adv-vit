export EPOCH=400 TRAIN_BATCH_SIZE=1024 WARMUP_EPOCH=5


python -u main_copy_fork_inc.py \
    --train-dataset-shards "$GCS_DATASET_DIR/cifar10-50m-wds/shards-{00000..00999}.tar" \
    --valid-dataset-shards  "$GCS_DATASET_DIR/cifar10-test-wds/shards-{00000..00099}.tar" \
    --train-origin-dataset-shards "$GCS_DATASET_DIR/cifar10-train-wds/shards-{00000..00099}.tar" \
    --layers 12  \
    --dim 768  \
    --heads 12  \
    --labels 10  \
    --patch-size 2  \
    --image-size 32  \
    --posemb "learnable"  \
    --pooling 'gap'  \
    --dropout 0.0  \
    --droppath 0.0  \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --learning-rate 1e-4 \
    --weight-decay 0.5 \
    --warmup-steps $((50000 * $WARMUP_EPOCH / $TRAIN_BATCH_SIZE)) \
    --training-steps $((50000 * $EPOCH / $TRAIN_BATCH_SIZE)) \
    --project cifar10-ablation-epoch \
    --name $(basename $0 .sh) \
    --output-dir "$GCS_DATASET_DIR/ablation/epoch" \
    --beta 5.0 \
    --label-smoothing 0.4 \
    --pretrained-ckpt "$GCS_DATASET_DIR/ablation/epoch/mae-ft-vit-b2-32-cifar10-400ep-batch-1024-ema.msgpack" \
    --ema-decay 0.9995 \
    --ipaddr $(curl -s ifconfig.me) \
    --hostname $(hostname) \