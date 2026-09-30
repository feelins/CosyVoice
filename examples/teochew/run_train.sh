#!/bin/bash
# 潮汕话 CosyVoice3 LLM 微调启动脚本（单卡）
# 前置: 数据已生成(data/train|dev.data.list), 依赖已装(pandas, deepspeed==0.15.1)
# 显存要求: >=24GB (12GB 会 OOM, 见 docs/teochew_finetune_guide.md)
set -e

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="/root/autodl-tmp/CosyVoice:/root/autodl-tmp/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

PRETRAIN="${PRETRAIN:-/root/autodl-tmp/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B}"
BIN="${BIN:-/root/autodl-tmp/envs/cosyvoice/bin}"

"${BIN}/torchrun" --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    /root/autodl-tmp/CosyVoice/cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config conf/cosyvoice3.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --qwen_pretrain_path "${PRETRAIN}/CosyVoice-BlankEN" \
    --onnx_path "${PRETRAIN}" \
    --model llm \
    --checkpoint "${PRETRAIN}/llm.pt" \
    --model_dir exp/cosyvoice3/llm/torch_ddp \
    --tensorboard_dir tensorboard/cosyvoice3/llm/torch_ddp \
    --ddp.dist_backend nccl \
    --num_workers 2 \
    --prefetch 100 \
    --pin_memory \
    --use_amp
