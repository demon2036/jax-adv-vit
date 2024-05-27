export EPOCH=2400 TRAIN_BATCH_SIZE=4096 WARMUP_EPOCH=5


python -u main_copy.py \
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
    --learning-rate 3e-4 \
    --weight-decay 0.2 \
    --warmup-steps $((1281167 * WARMUP_EPOCH / $TRAIN_BATCH_SIZE)) \
    --training-steps $((50000 * $EPOCH / $TRAIN_BATCH_SIZE)) \
    --project cifar10-20m \
    --name $(basename $0 .sh) \