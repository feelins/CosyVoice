#!/bin/bash
# 藏语 CosyVoice3 LLM 微调 —— 12GB 单卡版（复用潮汕话 ZeRO-2 offload 方案）
#
# 与潮汕话 run_train_12g.sh 唯一差别：模型/日志目录改为 tibetan；
# 数据已由 local/prepare_data.py 生成（data/train.data.list, data/dev.data.list）。
set -e

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="/root/autodl-tmp/CosyVoice:/root/autodl-tmp/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

PRETRAIN="${PRETRAIN:-/root/autodl-tmp/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B}"
BIN="${BIN:-/root/autodl-tmp/envs/cosyvoice/bin}"
DS_CONFIG="${DS_CONFIG:-conf/ds_zero2_offload.json}"

# DeepSpeedCPUAdam 是 JIT 编译的 C++ 扩展，torch 的 verify_ninja_availability()
# 直接找 PATH 里的 ninja，所以必须把 env 的 bin 放进 PATH。
export PATH="${BIN}:${PATH}"

"${BIN}/torchrun" --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    /root/autodl-tmp/CosyVoice/cosyvoice/bin/train.py \
    --train_engine deepspeed \
    --deepspeed_config "${DS_CONFIG}" \
    --config conf/cosyvoice3.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --qwen_pretrain_path "${PRETRAIN}/CosyVoice-BlankEN" \
    --onnx_path "${PRETRAIN}" \
    --model llm \
    --checkpoint "${PRETRAIN}/llm.pt" \
    --model_dir exp/tibetan/llm/torch_ddp \
    --tensorboard_dir tensorboard/tibetan/llm/torch_ddp \
    --ddp.dist_backend nccl \
    --num_workers 2 \
    --prefetch 50 \
    --pin_memory
