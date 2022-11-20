export CUDA_VISIBLE_DEVICES=$1
LOG_DIR=$2
DATASET=$3
LABELED_LIST=$4
PRETRAIN_CKPT=$5
AUG_PRETRAIN_CKPT=$6
BATCH_SIZE=$7
BOX_NUM=$8
BOX_NUM_UNLABELED=$9
mkdir -p "${LOG_DIR}";
python -u train.py --log_dir="${LOG_DIR}" --dataset="${DATASET}" --batch_size="${BATCH_SIZE}" \
--labeled_sample_list="${LABELED_LIST}" --detector_checkpoint="${PRETRAIN_CKPT}" --box_num="${BOX_NUM}" --box_num_unlableed="${BOX_NUM_UNLABELED}" \
--augmentor_checkpoint="${AUG_PRETRAIN_CKPT}" --view_stats \
2>&1|tee "${LOG_DIR}"/LOG.log &
