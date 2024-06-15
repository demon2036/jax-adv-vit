export train_batch_size=16384 warmup_epoch=80 epoch=800

python3 src/main_pretrain_mae.py \
    --output-dir $GCS_MODEL_DIR/mae \
    --train-dataset-shards "$GCS_DATASET_DIR/cifar10-50m-wds/shards-{00000..00999}.tar" \
    --train-batch-size $train_batch_size \
    --train-loader-workers 80 \
    --random-crop rrc \
    --color-jitter 0.0 \
    --random-erasing 0.0 \
    --augment-repeats 1 \
    --test-crop-ratio 1.0 \
    --label-smoothing 0.0 \
    --layers 12 \
    --dim 768 \
    --heads 12 \
    --labels 1000 \
    --layerscale \
    --patch-size 2 \
    --image-size 32 \
    --posemb learnable \
    --pooling gap \
    --dropout 0.0 \
    --droppath 0.0 \
    --init-seed 0 \
    --mixup-seed 0 \
    --dropout-seed 0 \
    --shuffle-seed 0 \
    --optimizer lamb \
    --learning-rate 4.8e-3 \
    --weight-decay 0.05 \
    --adam-b1 0.9 \
    --adam-b2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay 1.0 \
    --clip-grad 0.0 \
    --grad-accum 1 \
    --warmup-steps $((1281167 * $warmup_epoch / $train_batch_size)) \
    --training-steps $((1281167 * $epoch / $train_batch_size)) \
    --log-interval 100 \
    --eval-interval $((1281167 * 5 / $train_batch_size)) \
    --project deit3-jax-mae-cifar10 \
    --name $(basename $0 .sh) \
    --ipaddr $(curl -s ifconfig.me) \
    --hostname $(hostname)