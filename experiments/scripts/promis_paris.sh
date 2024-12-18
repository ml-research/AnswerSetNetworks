# Script to run the PROMIS experiment with different step sizes
# 64 128 256 512 1024 2048 4096 8192 16384 32768 

# For promis paris you need to download the npy files containing the cartography data.
# We put all data on Huggingface 
# https://huggingface.co/datasets/DanielOchs/AnswerSetNetworks
# download the bercyarena.npy,eifeltower.npy, embassy.npy, government.npy, park.npy, primary.npy and secondary.npy files and put them in the data/promis_paris folder
#TODO automatically download the data from Huggingface


source .env
echo "workspace dir ${WORKSPACE_DIR}"

size=6500
for steps in 250000
do
    python ${WORKSPACE_DIR}/experiments/promis_paris.py --title  "promis_paris_${size}x${size}_stepsize_${steps}" \
            --steps ${steps} \
            --size ${size} \
            --log-path "/${WORKSPACE_DIR}/experiments/logs/promis/paris/" \
            --data-path "${WORKSPACE_DIR}/data/promis_paris/"
done