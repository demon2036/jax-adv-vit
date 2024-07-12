export EPOCH=6000 TRAIN_BATCH_SIZE=1024 WARMUP_EPOCH=5


python  -u hw.py \
    --train-dataset-shards "$GCS_DATASET_DIR/cifar10-50m-wds/shards-{00000..00999}.tar" \
    --valid-dataset-shards  "$GCS_DATASET_DIR/cifar10-test-wds/shards-{00000..00099}.tar" \
    --train-origin-dataset-shards "$GCS_DATASET_DIR/cifar10-train-wds/shards-{00000..00099}.tar" \
    --layers 12  \
    --dim 768  \
    --heads 12  \
    --labels 10  \
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
    --eval-interval $((50000 * 50 / $TRAIN_BATCH_SIZE)) \
    --project cifar10-l2-ablation-beta \
    --name $(basename $0 .sh) \
    --output-dir "$GCS_DATASET_DIR/best/cifar10-l2" \
    --beta 3.0 \
    --label-smoothing 0.4 \
    --pretrained-ckpt  "$GCS_DATASET_DIR/best/cifar10-l2/vit-b2-32-cifar10-l2-6000ep-ls0.4-gap-beta3-ema"