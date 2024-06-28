export EPOCH=2000 TRAIN_BATCH_SIZE=16 WARMUP_EPOCH=5


python -u main_wideresnet.py \
    --train-dataset-shards "cifar10-train-wds/shards-{00000..00099}.tar" \
    --valid-dataset-shards  "cifar10-train-wds/shards-{00000..00099}.tar" \
    --train-origin-dataset-shards "cifar10-train-wds/shards-{00000..00099}.tar" \
    --layers 12  \
    --dim 768  \
    --heads 12  \
    --labels 10  \
    --layerscale   \
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
    --project cifar10-ablation-wrn \
    --name $(basename $0 .sh) \
    --output-dir "$GCS_DATASET_DIR/ablation/wrn" \
    --beta 5.0 \
    --label-smoothing 0.1 \
    --ema-decay 0.995