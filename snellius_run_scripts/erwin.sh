cd experiments

uv run train_shapenet.py \
    --num-epochs 100000 \
    --data-path "/scratch-shared/scur2588/data/mlcfd_data/preprocessed" \
    --batch-size 2 \
    --msa-type BallMSA \
    --nsa-type NSAMSA