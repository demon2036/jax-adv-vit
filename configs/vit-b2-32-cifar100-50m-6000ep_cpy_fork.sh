export EPOCH=6000 TRAIN_BATCH_SIZE=4096 WARMUP_EPOCH=5  GCS_DATASET_DIR=gs://fbs0_dl_bucket


python -u main_copy_fork.py \
    --train-dataset-shards "$GCS_DATASET_DIR/cifar100-50m-wds/shards-{00000..00999}.tar" \
    --valid-dataset-shards  "$GCS_DATASET_DIR/cifar100-test-wds/shards-{00000..00099}.tar" \
    --layers 12  \
    --dim 768  \
    --heads 12  \
    --labels 100  \
    --layerscale   \
    --patch-size 2  \
    --image-size 32  \
    --posemb "learnable"  \
    --pooling 'cls'  \
    --dropout 0.0  \
    --droppath 0.0  \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --learning-rate 4e-4 \
    --weight-decay 0.2 \
    --warmup-steps $((50000 * $WARMUP_EPOCH / $TRAIN_BATCH_SIZE)) \
    --training-steps $((50000 * $EPOCH / $TRAIN_BATCH_SIZE)) \
    --project cifar100-50m \
    --name $(basename $0 .sh) \