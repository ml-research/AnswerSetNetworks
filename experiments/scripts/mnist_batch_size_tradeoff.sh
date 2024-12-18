# TRADE-OFF EXPERIMENT: batchsize vs time
# run ASN with different batchsizes 64,128,256,...30k for 50 epochs
# collect time for each epoch and performance after each epoch
# do this for T1, T2, T3


#for batch_size in 64 128 256 512 1024 2048 4096 8192 15000 30000
# only run for 10, 20, 50 epochs for varying batchsizes as we do not need the full 50 epochs to reach 98% accuracy

source .env
echo "workspace dir ${WORKSPACE_DIR}"

for batch_size in 64 100 128 256 512
do
    for i in 2 # 3 4
    do
        for seed in 0 1 2 3 4
        do
            python ${WORKSPACE_DIR}/experiments/mnist_addition.py --title  "mnist_addition_trade_off_T${i-1}_seed_${seed}_batch_size_${batch_size}" \
                    --num-digits $i \
                    --learning-rate 0.005 \
                    --batch-size $batch_size \
                    --num-epochs 10 \
                    --num-runs 1 \
                    --seed $seed \
                    --device "cuda" \
                    --log-path "${WORKSPACE_DIR}/experiments/logs/trade_off/"
        done
    done
done

for batch_size in 1024 2048 
do
    for i in 2 # 3 4
    do
        for seed in 0 1 2 3 4
        do
            python ${WORKSPACE_DIR}/experiments/mnist_addition.py --title  "mnist_addition_trade_off_T${i-1}_seed_${seed}_batch_size_${batch_size}" \
                    --num-digits $i \
                    --learning-rate 0.005 \
                    --batch-size $batch_size \
                    --num-epochs 20 \
                    --num-runs 1 \
                    --seed $seed \
                    --device "cuda" \
                    --log-path "${WORKSPACE_DIR}/experiments/logs/trade_off/"
        done
    done
done

for batch_size in 4096 8192 15000 30000
do
    for i in 2 # 3 4
    do
        for seed in 0 1 2 3 4
        do
            python ${WORKSPACE_DIR}/experiments/mnist_addition.py --title  "mnist_addition_trade_off_T${i-1}_seed_${seed}_batch_size_${batch_size}" \
                    --num-digits $i \
                    --learning-rate 0.005 \
                    --batch-size $batch_size \
                    --num-epochs 50 \
                    --num-runs 1 \
                    --seed $seed \
                    --device "cuda" \
                    --log-path "${WORKSPACE_DIR}/experiments/logs/trade_off/"
        done
    done
done