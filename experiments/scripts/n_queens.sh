source .env
echo "workspace dir ${WORKSPACE_DIR}"

for nq in 4 5 6 
do
    python ${WORKSPACE_DIR}/experiments/n_queens.py --title  "n_queens_comparison_nq_${nq}" \
            --num-queens ${nq} \
            --device "cuda" \
            --log-path "${WORKSPACE_DIR}/experiments/queen_logs/"
done

