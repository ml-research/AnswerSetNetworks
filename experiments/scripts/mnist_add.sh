# run mnist addition

source .env
echo "workspace dir ${WORKSPACE_DIR}"

# T1
for batch_size in 512
do
    for seed in 42
    do
        python ${WORKSPACE_DIR}/experiments/mnist_addition.py --title  "mnist_addition_comparison_T1_seed_${seed}_batch_size_${batch_size}" \
                --num-digits 2 \
                --learning-rate 0.005 \
                --batch-size $batch_size \
                --num-epochs 10 \
                --num-runs 1 \
                --seed $seed \
                --device "cuda" \
                --log-path "${WORKSPACE_DIR}/experiments/logs/compare/"
    done
done
