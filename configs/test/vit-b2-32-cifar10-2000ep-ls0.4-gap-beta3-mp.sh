export EPOCH=6000 TRAIN_BATCH_SIZE=4096 WARMUP_EPOCH=100


python -u main_copy_fork.py \
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
    --weight-decay 0.0 \
    --adam-b1 0.9 \
    --adam-b2 0.99 \
    --clip-grad 10.0 \
    --warmup-steps $((50000 * $WARMUP_EPOCH / $TRAIN_BATCH_SIZE)) \
    --training-steps $((50000 * $EPOCH / $TRAIN_BATCH_SIZE)) \
    --eval-interval $((50000 * 50 / $TRAIN_BATCH_SIZE)) \
    --project cifar1000-ablation-beta-new \
    --name $(basename $0 .sh) \
    --output-dir "$GCS_DATASET_DIR/ablation/mp" \
    --beta 3.0 \
    --label-smoothing 0.4