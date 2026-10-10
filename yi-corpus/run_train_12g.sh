#!/bin/bash
# 彝语 CosyVoice3 LLM 微调 —— 12GB 单卡版（复用藏语 ZeRO-2 offload 方案）
#
# 与 examples/tibetan/run_train_12g.sh 唯一差别：模型/日志目录改为 yi；
# 数据已由 scripts/prepare_parquet.py 生成（data/train.data.list, data/dev.data.list）。
#
# 注意：本脚本“不要”在准备好之前执行；确认 parquet 已生成、显存/磁盘充足后再运行。
# 训练只微调 llm（flow/hift 冻结），推理须 text_frontend=False + instruct「请用彝语表达」。
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
    --model_dir exp/yi/llm/torch_ddp \
    --tensorboard_dir tensorboard/yi/llm/torch_ddp \
    --ddp.dist_backend nccl \
    --num_workers 1 \
    --prefetch 2
    # 说明：容器 cgroup 内存上限仅 30GB，之前 num_workers=2+prefetch=50+pin_memory
    # 把 speech_tokenizer/campplus 的 onnx（每 worker 一份）+ 大量缓冲 batch 顶满 30GB
    # 被 OOM killer(SIGKILL/-9) 杀掉。改单进程加载 + 低 prefetch + 关闭 pin_memory，
    # 把 CPU 内存压到 ~15GB 以下（offload 的优化器状态约 7GB 必须保留，否则 12GB 显存放不下）。
