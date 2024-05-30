export EPOCH=10000 TRAIN_BATCH_SIZE=1024 WARMUP_EPOCH=100  GCS_DATASET_DIR=gs://caster-us-central-2b


python -u main_copy.py \
    --train-dataset-shards "$GCS_DATASET_DIR/cifar10-50m-wds/shards-{00000..00806}.tar" \
    --valid-dataset-shards  "gs://caster-us-central-2b/cifar10-test-wds/shards-{00000..00078}.tar" \
    --layers 12  \
    --dim 768  \
    --heads 12  \
    --labels 10  \
    --layerscale   \
    --patch-size 2  \
    --image-size 32  \
    --posemb "learnable"  \
    --pooling 'cls'  \
    --dropout 0.0  \
    --droppath 0.0  \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --learning-rate 1e-4 \
    --weight-decay 0.5 \
    --warmup-steps $((50000 * $WARMUP_EPOCH / $TRAIN_BATCH_SIZE)) \
    --training-steps $((50000 * $EPOCH / $TRAIN_BATCH_SIZE)) \
    --project cifar10-20m \
    --name $(basename $0 .sh) \