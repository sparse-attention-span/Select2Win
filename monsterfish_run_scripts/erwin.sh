cd experiments

uv run train_shapenet.py \
    --num-epochs 100 \
    --data-path "$HOME/erwin/shapenet_car/preprocessed" \
    --use-wandb 0 \
    --batch-size 2 \
    --msa-type BallMSA