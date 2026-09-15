#!/usr/bin/env bash
# ==============================================================================
# Script đánh giá kết quả mô hình Neuro-Symbolic CBM bằng KMeans Rule Book
# File: run_eval_rulebook.sh
# ==============================================================================
# Usage:
#   bash run_eval_rulebook.sh [backbone] [modality] [checkpoint_path] [data_path]
#
# Ví dụ:
#   # Đánh giá HyperGCN Joint trên Kaggle:
#   bash run_eval_rulebook.sh hypergcn j work_dir/ntu_cs/hypergcn_joint_symbolic/joint_best.pt /kaggle/input/datasets/nguyenbaohuy05/ntu60csfull/NTU60_CS.npz
#
#   # Đánh giá SkateFormer Joint:
#   bash run_eval_rulebook.sh skateformer j work_dir/ntu_cs/skateformer_joint_symbolic/joint_best.pt /kaggle/input/datasets/nguyenbaohuy05/ntu60csfull/NTU60_CS.npz
# ==============================================================================

set -e

BACKBONE=${1:-"hypergcn"}
MODALITY=${2:-"j"}
DATA_PATH=${4:-"/kaggle/input/datasets/nguyenbaohuy05/ntu60csfull/NTU60_CS.npz"}

# Tự động suy ra config và checkpoint mặc định nếu không truyền vào
CONFIG_PATH="config/ntu/${BACKBONE}/ntu_cs_${MODALITY}.yaml"
CHECKPOINT_PATH=${3:-"work_dir/ntu_cs/${BACKBONE}_${MODALITY}_symbolic/joint_best.pt"}

# Kiểm tra nếu đường dẫn checkpoint mặc định dạng joint
if [ ! -f "$CHECKPOINT_PATH" ] && [ "$MODALITY" = "j" ]; then
    CHECKPOINT_PATH="work_dir/ntu_cs/${BACKBONE}_joint_symbolic/joint_best.pt"
fi

echo "=================================================================="
echo "  Chạy Đánh Giá KMeans Rule Book (eval_rulebook.py)"
echo "  - Backbone   : $BACKBONE"
echo "  - Modality   : $MODALITY"
echo "  - Config     : $CONFIG_PATH"
echo "  - Checkpoint : $CHECKPOINT_PATH"
echo "  - Dataset    : $DATA_PATH"
echo "=================================================================="

python eval_rulebook.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --data "$DATA_PATH" \
    --device 0 \
    --kmeans-k 3 \
    --max-per-class 100 \
    --sweep \
    "${@:5}"
