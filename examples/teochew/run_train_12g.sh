#!/bin/bash
# 潮汕话 CosyVoice3 LLM 微调 —— 12GB 单卡版（RTX 3080 Ti / 2080 Ti 等）
#
# 与 run_train.sh 的唯一差别：把 train_engine 从 torch_ddp 换成 deepspeed，
# 并挂上 conf/ds_zero2_offload.json（ZeRO-2 + 优化器状态 CPU offload）。
#
# 为什么必须这么做：0.5B 其实是 642M 参数，fp32 全量微调的
#   权重 2.57 + 梯度 2.57 + Adam m 2.57 + Adam v 2.57 = 10.28 GB
# 光优化器状态就超过 12GB 卡的可用量，所以「调小 batch」没用 ——
# 瓶颈在静态显存，不在激活值。offload 后显存只剩权重+梯度约 5.14GB。
#
# 内存要求：>= 16GB（offload 的 m/v 约 5.14GB 常驻内存，本机 251GB 绰绰有余）
set -e

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTHONIOENCODING=UTF-8
export PYTHONPATH="/root/autodl-tmp/CosyVoice:/root/autodl-tmp/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH}"

PRETRAIN="${PRETRAIN:-/root/autodl-tmp/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B}"
BIN="${BIN:-/root/autodl-tmp/envs/cosyvoice/bin}"
DS_CONFIG="${DS_CONFIG:-conf/ds_zero2_offload.json}"

# DeepSpeedCPUAdam 是 JIT 编译的 C++ 扩展，torch 的 verify_ninja_availability()
# 是直接找 PATH 里的 ninja 可执行文件，所以必须把 env 的 bin 放进 PATH。
export PATH="${BIN}:${PATH}"

# 注意：不要加 --use_amp。deepspeed 路径下 dtype 由 ds config 的 bf16 决定
# （见 cosyvoice/utils/train_utils.py:75-83），重复传反而会创建无用的 GradScaler。
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
    --model_dir exp/cosyvoice3/llm/torch_ddp \
    --tensorboard_dir tensorboard/cosyvoice3/llm/torch_ddp \
    --ddp.dist_backend nccl \
    --num_workers 8 \
    --prefetch 50 \
    --pin_memory
