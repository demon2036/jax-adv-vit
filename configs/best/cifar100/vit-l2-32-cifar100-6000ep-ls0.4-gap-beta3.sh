export EPOCH=6000 TRAIN_BATCH_SIZE=1024 WARMUP_EPOCH=5


python -u main_copy_fork.py \
    --train-dataset-shards "$GCS_DATASET_DIR/cifar100-50m-wds/shards-{00000..00999}.tar" \
    --valid-dataset-shards  "$GCS_DATASET_DIR/cifar100-test-wds/shards-{00000..00099}.tar" \
    --train-origin-dataset-shards "$GCS_DATASET_DIR/cifar100-train-wds/shards-{00000..00099}.tar" \
    --layers 24  \
    --dim 1024  \
    --heads 16  \
    --labels 100  \
    --layerscale   \
    --patch-size 2  \
    --image-size 32  \
    --posemb "learnable"  \
    --pooling gap  \
    --dropout 0.0  \
    --droppath 0.0  \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --learning-rate 1e-4 \
    --weight-decay 0.5 \
    --warmup-steps $((50000 * $WARMUP_EPOCH / $TRAIN_BATCH_SIZE)) \
    --training-steps $((50000 * $EPOCH / $TRAIN_BATCH_SIZE)) \
    --eval-interval $((50000 * 5 / $TRAIN_BATCH_SIZE)) \
    --project cifar100-50m-best \
    --name $(basename $0 .sh) \
    --output-dir "$GCS_DATASET_DIR/best/cifar100-50m" \
    --beta 3.0 \
    --label-smoothing 0.4