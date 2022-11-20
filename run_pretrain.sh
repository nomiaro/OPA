export CUDA_VISIBLE_DEVICES=$1
LOG_DIR=$2
DATASET=$3
LABELED_LIST=$4
BATCH_SIZE=$5
BOX_NUM=$6
LAMBDA=$7
WARM_UP=$8
mkdir -p "${LOG_DIR}"
python -u pretrain.py --log_dir="${LOG_DIR}" --dataset="${DATASET}" --warm_up="${WARM_UP}" \
--batch_size="${BATCH_SIZE}" --labeled_sample_list="${LABELED_LIST}" --box_num="${BOX_NUM}" --Lambda="${LAMBDA}" \
2>&1|tee "${LOG_DIR}"/LOG_ALL.log &
