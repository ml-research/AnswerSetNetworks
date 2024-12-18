# run mnist with 5 seeds with BS=100 and best performing batchsize for 20 epochs and collect time for each batch and performance after each batch 

source .env
echo "workspace dir ${WORKSPACE_DIR}"

# T1
for batch_size in 100 512
do
    for seed in 1 2 3 4 5
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

# T2
for batch_size in 100 1024
do
    for seed in 1 2 3 4 5
    do
        python ${WORKSPACE_DIR}/experiments/mnist_addition.py --title  "mnist_addition_comparison_T2_seed_${seed}_batch_size_${batch_size}" \
                --num-digits 3 \
                --learning-rate 0.005 \
                --batch-size $batch_size \
                --num-epochs 10 \
                --num-runs 1 \
                --seed $seed \
                --device "cuda" \
                --log-path "${WORKSPACE_DIR}/experiments/logs/compare/"
    done
done

# T3
for batch_size in 100 4096
do
    for seed in 1 2 3 4 5
    do
        python ${WORKSPACE_DIR}/experiments/mnist_addition.py --title  "mnist_addition_comparison_T3_seed_${seed}_batch_size_${batch_size}" \
                --num-digits 4 \
                --learning-rate 0.005 \
                --batch-size $batch_size \
                --num-epochs 10 \
                --num-runs 1 \
                --seed $seed \
                --device "cuda" \
                --log-path "${WORKSPACE_DIR}/experiments/logs/compare/"
    done
done