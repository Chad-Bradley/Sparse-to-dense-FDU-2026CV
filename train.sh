#!/usr/bin/env bash
# s2d_baseline 的基础训练脚本。
#
# 默认就是「忠实复现」配置：
#   ResNet50 + upproj + rgbd + 500 点
#   ImageNet 归一化 + Lighting，15 epoch，batch 8，lr 0.01（每 5 epoch ×0.1）
#
# 默认关闭结构改进；可用环境变量或末尾 CLI 参数开启。
#
# 用法（在 s2d_baseline 目录或任意目录）：
#
#   bash train.sh
#   GPU=3 bash train.sh                      # 指定卡
#   EPOCHS=1 WORKERS=0 bash train.sh         # 快速试跑
#   DECODER=deconv3 bash train.sh            # 换成 deconv3
#   SAMPLES=200 bash train.sh                # 200 点对照
#   RESUME=runs/<run>/last.pth.tar bash train.sh
#
# 环境变量：
#   GPU CONDA_ENV ARCH DECODER MODALITY SAMPLES EPOCHS BATCH_SIZE LR
#   WORKERS CRITERION SEED PRETRAIN METRIC DATA DATA_ROOT S2D_DATA_ROOT
#   TORCH_HOME S2D_RUNS RUN_NAME RESUME LIMIT_TRAIN LIMIT_VAL PRINT_FREQ
#
# 数据路径优先级：DATA_ROOT > S2D_DATA_ROOT > <repo>/data/<DATA>。
# 本仓库已从大仓库拆出，数据可以用软链接或环境变量指过去，例如：
#   ln -s /root/Test/data data
#   # 或
#   export S2D_DATA_ROOT=/root/DZQ/CV/sparse-to-dense.pytorch-master/data/nyudepthv2
#
# 额外参数会原样追加到 python 命令末尾，例如：
#   bash train.sh --epochs 2 --workers 0

set -euo pipefail

S2D_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$S2D_DIR"

# --- 默认设置 --------------------------------------------------------------
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
CONDA_ENV="${CONDA_ENV:-s2d}"
ARCH="${ARCH:-resnet50}"
DECODER="${DECODER:-upproj}"
MODALITY="${MODALITY:-rgbd}"
SAMPLES="${SAMPLES:-500}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-0.01}"
WORKERS="${WORKERS:-10}"
CRITERION="${CRITERION:-l1}"
SEED="${SEED:-0}"
PRETRAIN="${PRETRAIN:-1}"
METRIC="${METRIC:-image}"
DATA="${DATA:-nyudepthv2}"
DATA_ROOT="${DATA_ROOT:-${S2D_DATA_ROOT:-$S2D_DIR/data/$DATA}}"
TORCH_HOME="${TORCH_HOME:-/root/Test/hf_cache}"
S2D_RUNS="${S2D_RUNS:-/root/Test/s2d_baseline_runs}"
RUN_NAME="${RUN_NAME:-}"
RESUME="${RESUME:-}"
MULTISCALE="${MULTISCALE:-none}"
UNET_SKIPS="${UNET_SKIPS:-0}"
REFINER="${REFINER:-none}"
PROP_KERNEL="${PROP_KERNEL:-3}"
PROP_STEPS="${PROP_STEPS:-3}"
PRESERVE_SPARSE="${PRESERVE_SPARSE:-1}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
REFINER_ONLY_EPOCHS="${REFINER_ONLY_EPOCHS:-3}"
REFINER_LR="${REFINER_LR:-0.001}"
JOINT_LR="${JOINT_LR:-0.0001}"
VAL_SAMPLING_SEED="${VAL_SAMPLING_SEED:-0}"
LIMIT_TRAIN="${LIMIT_TRAIN:-0}"
LIMIT_VAL="${LIMIT_VAL:-0}"
PRINT_FREQ="${PRINT_FREQ:-50}"
LOG_DIR="${LOG_DIR:-$S2D_RUNS/logs}"

export CUDA_VISIBLE_DEVICES="$GPU"
export TORCH_HOME
export S2D_RUNS
export PYTHONUNBUFFERED=1

# --- conda -----------------------------------------------------------------
find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi
    local candidate
    for candidate in "${CONDA_EXE:-}" \
                     /opt/conda/bin/conda \
                     "${HOME:-/root}/miniconda3/bin/conda" \
                     "${HOME:-/root}/anaconda3/bin/conda" \
                     /usr/local/miniconda3/bin/conda; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if ! CONDA_BIN="$(find_conda)"; then
    echo "✗ 找不到 conda；可手动指定 CONDA_EXE=/path/to/conda" >&2
    exit 1
fi
CONDA_BASE="$("$CONDA_BIN" info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- 检查 ------------------------------------------------------------------
if [[ ! -e "$DATA_ROOT/train" ]]; then
    echo "✗ 找不到 $DATA_ROOT/train" >&2
    echo "  检查仓库根目录的 data 软链接是否指向 /root/Test/data" >&2
    exit 1
fi
if [[ ! -d "$TORCH_HOME/hub/checkpoints" ]]; then
    echo "! $TORCH_HOME/hub/checkpoints 不存在，预训练权重可能联网下载" >&2
fi
if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "✗ CUDA 不可用；检查 CUDA_VISIBLE_DEVICES / torch 是否是 cu121 构建" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -n "$RUN_NAME" ]]; then
    LOG_FILE="$LOG_DIR/baseline_${RUN_NAME}_${STAMP}.log"
else
    LOG_FILE="$LOG_DIR/baseline_${ARCH}_${DECODER}_${MODALITY}_s${SAMPLES}_${STAMP}.log"
fi

# --- 组装命令 --------------------------------------------------------------
CMD=(python -u -m scripts.train
     -a "$ARCH" -d "$DECODER" -m "$MODALITY" -s "$SAMPLES"
     --data-root "$DATA_ROOT"
     --epochs "$EPOCHS"
     --batch-size "$BATCH_SIZE"
     --lr "$LR"
     --criterion "$CRITERION"
     --metric-reduction "$METRIC"
     --workers "$WORKERS"
     --seed "$SEED"
     --val-sampling-seed "$VAL_SAMPLING_SEED"
     --multiscale "$MULTISCALE"
     --refiner "$REFINER"
     --prop-kernel "$PROP_KERNEL"
     --prop-steps "$PROP_STEPS"
     --refiner-only-epochs "$REFINER_ONLY_EPOCHS"
     --refiner-lr "$REFINER_LR"
     --joint-lr "$JOINT_LR"
     --print-freq "$PRINT_FREQ"
     --out-dir "$S2D_RUNS")

if [[ "$PRETRAIN" != "1" ]]; then
    CMD+=(--no-pretrain)
fi
if [[ -n "$RUN_NAME" ]]; then
    CMD+=(--run-name "$RUN_NAME")
fi
if [[ -n "$RESUME" ]]; then
    CMD+=(--resume "$RESUME")
fi
if [[ -n "$INIT_CHECKPOINT" ]]; then
    CMD+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [[ "$UNET_SKIPS" == "1" ]]; then
    CMD+=(--unet-skips)
fi
if [[ "$PRESERVE_SPARSE" == "0" ]]; then
    CMD+=(--no-preserve-sparse)
fi
if [[ "$LIMIT_TRAIN" -gt 0 ]]; then
    CMD+=(--limit-train "$LIMIT_TRAIN")
fi
if [[ "$LIMIT_VAL" -gt 0 ]]; then
    CMD+=(--limit-val "$LIMIT_VAL")
fi
if [[ "$#" -gt 0 ]]; then
    CMD+=("$@")
fi

# --- 启动 ------------------------------------------------------------------
echo "=> s2d_baseline 训练"
echo "   python     $(command -v python)"
echo "   torch      $(python -c 'import torch; print(torch.__version__, "cuda=" + str(torch.cuda.is_available()))')"
echo "   GPU        $CUDA_VISIBLE_DEVICES"
echo "   数据       $DATA_ROOT"
echo "   输出       $S2D_RUNS"
echo "   日志       $LOG_FILE"
echo "   命令       ${CMD[*]}"
echo "----------------------------------------------------------------------"

if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
    status=0
else
    status=$?
fi

echo "----------------------------------------------------------------------"
echo "=> 退出码 $status"
echo "   日志 $LOG_FILE"
exit "$status"
