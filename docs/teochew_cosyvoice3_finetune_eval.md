# 潮汕话 CosyVoice3 微调 & 验证 完整指南

> 记录本次在 **RTX 3080 Ti 12GB 单卡** 上微调 Fun-CosyVoice3-0.5B（潮汕话/闽南语系）的全过程：
> 训练方案、踩过的坑、诊断方法，以及**训后如何验证合成效果**的推理脚本。
> 时间线：2026-09-30 开始；2026-10-01 训完 20 epoch 并实测推理有效。

---

## 0. 一句话结论

- 12GB 单卡能跑通 0.5B 全量微调，**必须用 DeepSpeed ZeRO-2 + 优化器 CPU offload**（否则静态显存超限）。
- 这台机器**内存 cgroup 上限是 30GB**（不是看起来的 251GB 宿主内存）。`num_workers` 必须设 **2**，设 8 会撑爆内存被 OOM Killer 直接 SIGKILL。
- 验证效果**靠人工主观 A/B 试听**（原版 vs 微调版），**不依赖自动评测**（自动评测不可靠）。
- **实测结论**：20 epoch 微调后，主观 A/B 对比明显——预训练版大部分按普通话读、无潮汕味；微调版有清晰潮汕话味道、发音非普通话。验证流程见第 5/6/7 节（含推理踩坑 6.4）。

---

## 1. 机器配置与隐藏约束

| 项 | 值 |
|---|---|
| GPU | RTX 3080 Ti 12GB × 1 |
| CPU | 10 vCPU (Xeon Gold 6248 @ 2.50GHz) |
| 内存 | 规格写 30GB；**cgroup v2 上限实测 `memory.max = 32212254720`（= 30GB）** |
| 系统 | PyTorch 2.5.1 / Python 3.10(conda) / CUDA 12.4 / Ubuntu |
| conda 环境 | `/root/autodl-tmp/envs/cosyvoice` |
| 项目根 | `/root/autodl-tmp/CosyVoice` |

**关键坑**：`free -g` 看到的 251GB 是宿主总内存，容器被 cgroup 限制在 **30GB**。
任何让进程 RSS 超过 30GB 的操作都会被内核 OOM Killer 杀掉（发 `SIGKILL`，**没有 Python 报错栈、也没有 CUDA OOM 提示**）。

```bash
# 查看本机内存上限（cgroup v2）
cat /sys/fs/cgroup/memory.max          # 输出 32212254720 = 30GB
# 实时查看当前 cgroup 用量（用来判断是否逼近上限）
cat /sys/fs/cgroup/memory.current
```

---

## 2. 训练方案（12GB 单卡版）

### 2.1 为什么必须用 DeepSpeed ZeRO-2 + CPU offload

0.5B 实际是 642M 参数。fp32 全量微调的静态显存：

```
权重 2.57 + 梯度 2.57 + Adam m 2.57 + Adam v 2.57 = 10.28 GB
```

仅优化器状态（m/v）就 ≈5.14GB，超过 12GB 卡的可用量。**调小 batch 没用 —— 瓶颈在静态显存，不在激活值。**

解决：ZeRO-2 把 Adam 的 m/v offload 到 CPU，显存只剩 权重+梯度 ≈5.14GB。

### 2.2 训练脚本 `examples/teochew/run_train_12g.sh`

```bash
#!/bin/bash
# 潮汕话 CosyVoice3 LLM 微调 —— 12GB 单卡版（RTX 3080 Ti / 2080 Ti 等）
#
# 与 run_train.sh 的唯一差别：把 train_engine 从 torch_ddp 换成 deepspeed，
# 并挂上 conf/ds_zero2_offload.json（ZeRO-2 + 优化器状态 CPU offload）。
#
# 为什么必须这么做：见上面「静态显存」分析，调小 batch 无效，瓶颈在静态显存。
# 内存要求：>= 16GB（offload 的 m/v 约 5.14GB 常驻内存，本机 cgroup 30GB 够用）。
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
# （见 cosyvoice/utils/train_utils.py），重复传反而会创建无用的 GradScaler。
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
    --num_workers 2 \
    --prefetch 50 \
    --pin_memory
```

### 2.3 DeepSpeed 配置 `examples/teochew/conf/ds_zero2_offload.json`

```json
{
    "comment": "12GB 单卡（RTX 3080 Ti）专用：ZeRO-2 + 优化器状态 CPU offload。把 Adam 的 m/v（约 5.14GB）挪到内存，显存只留权重+梯度（约 5.14GB）。本机内存 cgroup 30GB，够用。",
    "train_batch_size": 2,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 5,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 500000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": true
    },
    "bf16": {
        "enabled": true
    },
    "steps_per_print": 50,
    "wall_clock_breakdown": false
}
```

> 注：`gradient_accumulation_steps=2` 让有效 batch 翻倍；日志里会打印
> `using accumulate grad, new batch size is 2 times larger than before`。

### 2.4 启动 & 日志

```bash
cd /root/autodl-tmp/CosyVoice/examples/teochew
nohup bash run_train_12g.sh > /root/autodl-tmp/CosyVoice/train_12g.log 2>&1 &
```

- **日志放在项目目录**：`/root/autodl-tmp/CosyVoice/train_12g.log`（不要放 `/tmp`）。
- 数据准备：`data/train.data.list` 与 `data/dev.data.list` 需事先准备好（训练前已就绪）。
  - 注意：这两个 list 不是明文文本，而是指向 parquet 的路径
    （如 `data/train/parquet/parquet_000000000.tar`，实际是 Apache Parquet，扩展名 `.tar` 而已）。
    parquet 字段：`utt / audio_data / wav / text(潮汕话句子) / spk / instruct`。
- 检查点：训练时落盘到 `exp/cosyvoice3/llm/torch_ddp/epoch_N_whole/`（DeepSpeed ZeRO 格式）。
- `latest` 文件：`exp/cosyvoice3/llm/torch_ddp/latest`，内容是最新 epoch 的 tag（如 `epoch_19_whole`）。
- TensorBoard：`tensorboard/cosyvoice3/llm/torch_ddp/`，可用 6006 端口查看。

### 2.5 启动后确认

训练正常启动会看到：

```
[Rank 0] Checkpoint init is about to be saved!
...
start step 0 start epoch -1
Epoch 0 TRAIN info lr 1e-05 rank 0
using accumulate grad, new batch size is 2 times larger than before
```

---

## 3. 踩坑记录（重要）

### 3.1 现象：`num_workers=8` 训练 ~2 分钟被 SIGKILL

首个 epoch 跑到 `Epoch 0 TRAIN` 约 2 分 19 秒后，进程被 `Signal 9 (SIGKILL)` 杀死，
报错只有：

```
torch.distributed.elastic.multiprocessing.errors.ChildFailedError
exitcode  : -9 (pid: xxxx)
traceback : Signal 9 (SIGKILL) received by PID xxxx
```

**没有 Python 栈，也没有 CUDA out-of-memory 提示。**

### 3.2 根因：撑爆 30GB 内存 cgroup → OOM Killer

诊断链路：
1. `/sys/fs/cgroup/memory.max` = `32212254720`（30GB）—— 这是容器真正的内存天花板。
2. `num_workers=8` 时，8 个 dataloader worker 进程 + 每个 worker 内的 ONNX CPU 会话
   + 预取缓冲（`prefetch 50`）会把容器 RSS 推过 30GB。
3. 内核 OOM Killer 直接 SIGKILL 训练进程（这就是"无栈、无 CUDA OOM"的原因）。
4. `num_workers=2` 不超 30GB，可稳定跑完多个 epoch（本次训满 20 epoch）。

### 3.3 结论

> **这台 12GB / 30GB 内存的机器，训练固定用 `num_workers=2`。不要再改大。**

`--num_workers` 是启动失败的唯一差异参数，与模型 / batch / 显存无关。

---

## 4. 训练监控

```bash
# GPU 显存 / 利用率
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# cgroup 内存用量（判断是否逼近 30GB 上限）
cat /sys/fs/cgroup/memory.current

# 训练日志尾部
tail -f /root/autodl-tmp/CosyVoice/train_12g.log

# 进程存活
pgrep -f "cosyvoice/bin/train.py" && echo 活着 || echo 已退出
```

---

## 5. 验证合成效果（推理）方案

### 5.1 总原则

- **主观试听为王，自动评测不可靠** —— 不采用 WER / 相似度 / UTMOS 等自动打分。
- 核心做法：**同一句文本 + 同一段参考音色**，分别在「原版预训练」和「微调版」上合成，
  成对地听，对比方言自然度 / 正确度 / 音色一致性。
- **实测结果（20 epoch）**：预训练版大部分按普通话读、几乎无潮汕味；微调版有清晰潮汕话味道、发音非普通话。微调确实有效。

### 5.2 CosyVoice3 推理接口（已核实）

- 加载：`from cosyvoice.cli.cosyvoice import AutoModel` → `AutoModel(model_dir=...)`，
  会根据目录里的 `cosyvoice3.yaml` 自动选 `CosyVoice3` 类。
- 方言合成（核心）：`cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=False, text_frontend=False)`
  - `instruct_text` **必须含 `<|endofprompt|>` 标记**（如 `"请用潮汕话表达。<|endofprompt|>"`，
    或训练数据里的 `"You are a helpful assistant. 请用潮汕话表达。<|endofprompt|>"`），
    否则推理端断言 `<|endofprompt|> not detected` 直接失败。方言内容由 LLM 决定，音色由 `prompt_wav` 决定。
  - `text_frontend=False`：**方言/潮汕话文本必须关掉前端中文归一化**，否则可能产出异常短的输出（见 6.4）。
- 音色克隆 / 管线 sanity check：`cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False, text_frontend=False)`
  - `prompt_text` 必须是参考音频的**真实转写**。
- 推理加载要求：`model_dir` 下需有 `llm.pt` / `flow.pt` / `hift.pt` / `cosyvoice3.yaml` /
  `campplus.onnx` / `speech_tokenizer_v3.onnx` / `CosyVoice-BlankEN/`。
  其中 `llm.pt` 必须是 **LLM 子模块的 state_dict**（`model.load` 用 `strict=True` 加载）。

### 5.3 微调权重如何变成可推理的 `llm.pt`

训练保存的是 DeepSpeed ZeRO 整体检查点（`epoch_N_whole/` 目录，内含 llm+flow+hift 全模型），
不能直接给推理用。需要：

1. 用 DeepSpeed 的 `get_fp32_state_dict_from_zero_checkpoint` 还原 fp32 全模型权重；
   **注意**：该函数要传**含 `latest` 文件的父目录**（`torch_ddp/`），不是 `epoch_N_whole/` 子目录，
   它会读 `latest` 自动定位最新 epoch 的 tag。
2. 抽取 LLM 部分权重。注意预训练 `llm.pt` 的 key **本身带 `llm.` 前缀**，且还含顶层
   `llm_decoder.weight` / `speech_embedding.weight`——所以**直接取「全模型键 ∩ 预训练 llm.pt 键」**
   （保留前缀）最稳，不要手动去前缀。
3. 组装一个"微调版模型目录"：其余文件（flow.pt / hift.pt / *.onnx / yaml / CosyVoice-BlankEN）
   全部软链接到预训练目录，只把上面得到的 `llm.pt` 写进去。

这一步由 `teochew_eval/convert_ckpt.py` 完成（见第 6.1 节，含转换踩坑）。

---

## 6. 推理脚本（已落地在 `teochew_eval/`）

### 6.1 `teochew_eval/convert_ckpt.py` —— 检查点转推理模型

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 CosyVoice3 DeepSpeed 训练产出的 ZeRO 整体检查点，转成推理可用的标准 llm.pt，
并组装一个「微调版模型目录」，供 eval_teochew.py / webui.py / infer_teochew_real.py 直接加载。

为什么需要这一步:
  训练保存的是 DeepSpeed ZeRO 整体检查点(epoch_N_whole/ 目录, 内含 llm+flow+hift 全模型)。
  而推理端 CosyVoice3.load 要求 llm.pt 是「LLM 子模块」的 state_dict(strict=True),
  所以这里用 deepspeed 的 get_fp32_state_dict_from_zero_checkpoint 还原 fp32 权重,
  再抽取 LLM 部分的权重。

用法(训练结束后执行):
  cd /root/autodl-tmp/CosyVoice
  python teochew_eval/convert_ckpt.py \
      --exp_dir examples/teochew/exp/cosyvoice3/llm/torch_ddp \
      --pretrained pretrained_models/Fun-CosyVoice3-0.5B \
      --out pretrained_models/Fun-CosyVoice3-0.5B-teochew

说明:
  --exp_dir 必须是「含 latest 文件的父目录」(torch_ddp/), 脚本读 latest 自动选最新 epoch。
  --out 目录里除 llm.pt 外, 其余文件(flow.pt / hift.pt / *.onnx / cosyvoice3.yaml /
  CosyVoice-BlankEN / campplus.onnx 等)全部用软链接指向预训练目录, 几乎不占额外空间。
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, 'third_party/Matcha-TTS')

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def get_llm_expected_keys(pretrained):
    # 直接用预训练 llm.pt 的 key 集合作为 LLM 期望结构(与微调版架构一致, 无需重建模型)
    sd = torch.load(os.path.join(pretrained, 'llm.pt'),
                    map_location='cpu', weights_only=True)
    return set(sd.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', default='examples/teochew/exp/cosyvoice3/llm/torch_ddp')
    ap.add_argument('--pretrained', default='pretrained_models/Fun-CosyVoice3-0.5B')
    ap.add_argument('--out', default='pretrained_models/Fun-CosyVoice3-0.5B-teochew')
    args = ap.parse_args()

    # get_fp32_state_dict_from_zero_checkpoint 需要「含 latest 文件的父目录」,
    # 它会读 latest 自动定位到具体 epoch 的 tag 子目录。
    latest_file = os.path.join(args.exp_dir, 'latest')
    tag = open(latest_file).read().strip() if os.path.exists(latest_file) else '?'
    print('latest ->', tag)
    print('正在还原 fp32 权重(CPU, 可能要 1~2 分钟):', args.exp_dir)
    full_sd = get_fp32_state_dict_from_zero_checkpoint(args.exp_dir)

    expected = get_llm_expected_keys(args.pretrained)
    # 预训练 llm.pt 的 key 既包含 'llm.' 前缀的子模块, 也包含顶层的
    # llm_decoder.weight / speech_embedding.weight, 故直接取「全模型键 ∩ 预训练键」最稳妥。
    llm_sd = {k: v for k, v in full_sd.items() if k in expected}
    missing = expected - set(llm_sd.keys())
    extra = set(llm_sd.keys()) - expected
    assert not missing, 'LLM 权重缺失: {}'.format(sorted(missing)[:10])
    assert not extra, 'LLM 有多余键: {}'.format(sorted(extra)[:10])
    print('LLM 权重对齐 OK, 共 {} 个参数'.format(len(llm_sd)))

    # 组装 out 目录(非 llm 文件全部软链)
    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.pretrained):
        src = os.path.abspath(os.path.join(args.pretrained, name))
        dst = os.path.join(args.out, name)
        if name in ('llm.pt', 'llm.rl.pt'):
            continue  # 这两个我们用微调版替换
        if os.path.islink(dst) or os.path.exists(dst):
            continue
        os.symlink(src, dst)
    torch.save(llm_sd, os.path.join(args.out, 'llm.pt'))
    print('已写出:', os.path.join(args.out, 'llm.pt'))
    print('完成! 随后可执行:')
    print('  python teochew_eval/infer_teochew_real.py --model_dir {}'.format(args.out))


if __name__ == '__main__':
    main()
```

**转换踩坑（实测）**：
- `get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)` 的 `ckpt_dir` 必须是**含 `latest` 的父目录**
  （`torch_ddp/`），传 `epoch_N_whole/` 子目录会报 `Unable to find 'latest' file`。
- 不要用 hyperpyyaml 重建 LLM 来拿 key 集合——`cosyvoice3.yaml` 里 `!new:cosyvoice.llm.llm.CosyVoice3LM`
  在脚本里会因模块未导入而报 `ImportError`。**直接拿预训练 `llm.pt` 的 key 集合当模板**更稳。
- 预训练 `llm.pt` 的 key **带 `llm.` 前缀**且含顶层 `llm_decoder.weight` / `speech_embedding.weight`，
  手动去前缀会漏掉这两键 → 断言 `LLM 权重缺失`。用「全模型键 ∩ 预训练键」取交集即可。

### 6.2 `teochew_eval/eval_teochew.py` —— 手工句子合成 & A/B 对比

适合放自己手写的潮汕话句子做验证（句子在脚本里 `TEochew_INSTRUCT` 列表里改）。

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""潮汕话(CosyVoice3 微调)效果验证: 合成样本 + (可选)与原版 A/B 对比。主观试听为主。

用法:
  cd /root/autodl-tmp/CosyVoice
  python teochew_eval/eval_teochew.py --compare --out teochew_eval/out
  python teochew_eval/eval_teochew.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B-teochew

注意:
  - 句子在 TEochew_INSTRUCT 里改(换成你自己的真实潮汕话语料)。
  - instruct 必须含 <|endofprompt|>; 方言文本要 text_frontend=False。
  - --prompt 默认 asset/zero_shot_prompt.wav(通用音色), 想验证潮汕音色请换成本地潮汕人声。
"""
import os
import sys
import argparse

ROOT = '/root/autodl-tmp/CosyVoice'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party/Matcha-TTS'))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

DEFAULT_PROMPT = './asset/zero_shot_prompt.wav'
# instruct 必须带 <|endofprompt|>, 否则推理断言失败
INSTRUCT = 'You are a helpful assistant. 请用潮汕话表达。<|endofprompt|>'

# ===== 把下面的句子换成你自己的潮汕话语料 =====
TEochew_INSTRUCT = [
    '落雨哩，路咧澹澹，行路爱细腻。',
    '伊昨昏去街路买一尾鱼转来。',
    '恁食饱未？行出去行行仔。',
    '这领衫诚雅，偌多钱？',
]
# 中文 zero-shot：验证管线 & 音色克隆（prompt_text 需与参考音频转写一致）
ZH_ZEROSHOT = [
    ('欢迎使用潮汕话语音合成，这是一段测试语音。',
     '希望你以后能够做的比我还好呦。'),
]


def synth_one(cosyvoice, out_dir, prefix, prompt_wav):
    os.makedirs(out_dir, exist_ok=True)
    for i, txt in enumerate(TEochew_INSTRUCT):
        for out in cosyvoice.inference_instruct2(
                txt, INSTRUCT, prompt_wav, stream=False, text_frontend=False):
            p = os.path.join(out_dir, '{}_teochew_{}.wav'.format(prefix, i))
            torchaudio.save(p, out['tts_speech'], cosyvoice.sample_rate)
            print('saved', p, flush=True)
    for i, (txt, prompt_text) in enumerate(ZH_ZEROSHOT):
        for out in cosyvoice.inference_zero_shot(
                txt, prompt_text, prompt_wav, stream=False, text_frontend=False):
            p = os.path.join(out_dir, '{}_zh_{}.wav'.format(prefix, i))
            torchaudio.save(p, out['tts_speech'], cosyvoice.sample_rate)
            print('saved', p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    ap.add_argument('--out', default='teochew_eval/out')
    ap.add_argument('--prompt', default=DEFAULT_PROMPT)
    ap.add_argument('--compare', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.prompt):
        print('!! 参考音频不存在:', args.prompt)
        sys.exit(1)

    if args.compare:
        print('加载原版...', flush=True)
        m0 = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
        print('加载微调版...', flush=True)
        m1 = AutoModel(model_dir=args.model_dir)
        synth_one(m0, os.path.join(args.out, 'pretrained'), 'base', args.prompt)
        synth_one(m1, os.path.join(args.out, 'finetuned'), 'ft', args.prompt)
    else:
        print('加载模型:', args.model_dir, flush=True)
        m = AutoModel(model_dir=args.model_dir)
        synth_one(m, args.out, 'ft', args.prompt)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
```

### 6.3 `teochew_eval/infer_teochew_real.py` —— 从训练集 parquet 抽真实句子（推荐）

实测用的就是它：从训练集 parquet 直接抽一条**原生说话人音色**（chao_shan_01）+ 若干条**真实潮汕话句子**，
自动做原版 vs 微调版 A/B，并写出 `manifest.txt` 对照清单。无需手写句子。

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从训练集 parquet 抽取真实潮汕话句子 + 原生说话人音色，做推理合成。
同时支持 --compare 在原版 / 微调版上各合成一遍，便于 A/B 试听。

数据来源: examples/teochew/data/train/parquet/parquet_000000000.tar
  字段: text(潮汕话句子) / instruct(请用潮汕话表达。) / audio_data(原生说话人 wav 字节)

用法:
  cd /root/autodl-tmp/CosyVoice
  # 只生成微调版
  python teochew_eval/infer_teochew_real.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B-teochew
  # 原版 vs 微调版 A/B
  python teochew_eval/infer_teochew_real.py --compare
"""
import os
import sys
import argparse

ROOT = '/root/autodl-tmp/CosyVoice'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party/Matcha-TTS'))

import torch
import torchaudio
import pyarrow.parquet as pq
from cosyvoice.cli.cosyvoice import AutoModel

PARQUET = 'examples/teochew/data/train/parquet/parquet_000000000.tar'
# 注意: instruct 必须带 <|endofprompt|> 标记(CosyVoice3 的硬性要求), 否则推理断言失败
INSTRUCT = 'You are a helpful assistant. 请用潮汕话表达。<|endofprompt|>'
PROMPT_PATH = 'teochew_eval/prompt_chaoshan.wav'


def pick_samples(n_text=8):
    t = pq.read_table(PARQUET)
    texts = t.column('text').to_pylist()
    audios = t.column('audio_data').to_pylist()
    # 过滤过长/过短, 并剔除含罕见字符(>U+FFFF, 如 𠁞)的句子, 避免分词后输出过短打不过 vocoder
    cand = [(tx, au) for tx, au in zip(texts, audios)
            if 4 <= len(str(tx)) <= 36
            and all(ord(c) < 0xFFFF for c in str(tx))]
    step = max(1, len(cand) // (n_text + 1))
    sel = cand[::step][: n_text + 1]
    prompt_audio = sel[0][1]          # 原生说话人音色
    target_texts = [s[0] for s in sel[1:n_text + 1]]
    return prompt_audio, target_texts


def synth(cosyvoice, out_dir, prefix, prompt_path, target_texts):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, tx in enumerate(target_texts):
        try:
            # text_frontend=False: 潮汕话/方言文本不做中文归一化, 否则可能产出异常短的输出
            for out in cosyvoice.inference_instruct2(
                    tx, INSTRUCT, prompt_path, stream=False, text_frontend=False):
                p = os.path.join(out_dir, '{}_teochew_{}.wav'.format(prefix, i))
                torchaudio.save(p, out['tts_speech'], cosyvoice.sample_rate)
                print('saved', p, flush=True)
                manifest.append((p, tx))
        except Exception as e:
            print('!! 第 {} 句合成失败: {} | 文本: {}'.format(i, repr(e), tx), flush=True)
    # 保存文本清单，方便对照试听
    with open(os.path.join(out_dir, 'manifest.txt'), 'w', encoding='utf-8') as f:
        for p, tx in manifest:
            f.write('{}\t{}\n'.format(os.path.basename(p), tx))
    print('清单已写', os.path.join(out_dir, 'manifest.txt'), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir',
                    default='pretrained_models/Fun-CosyVoice3-0.5B-teochew')
    ap.add_argument('--out', default='teochew_eval/out')
    ap.add_argument('--n_text', type=int, default=8)
    ap.add_argument('--compare', action='store_true')
    args = ap.parse_args()

    prompt_audio, target_texts = pick_samples(args.n_text)
    with open(PROMPT_PATH, 'wb') as f:
        f.write(prompt_audio)
    print('参考音色:', PROMPT_PATH, '| 目标潮汕话句数:', len(target_texts), flush=True)
    for tx in target_texts:
        print('  -', tx, flush=True)

    if args.compare:
        print('加载原版...', flush=True)
        synth(AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B'),
              os.path.join(args.out, 'pretrained'), 'base',
              PROMPT_PATH, target_texts)
        print('加载微调版...', flush=True)
        synth(AutoModel(model_dir=args.model_dir),
              os.path.join(args.out, 'finetuned'), 'ft',
              PROMPT_PATH, target_texts)
    else:
        print('加载微调版:', args.model_dir, flush=True)
        synth(AutoModel(model_dir=args.model_dir),
              args.out, 'ft', PROMPT_PATH, target_texts)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
```

### 6.4 推理阶段踩坑（实测）

1. **`ModuleNotFoundError: No module named 'cosyvoice'`**
   用 `python teochew_eval/xxx.py` 跑时，`sys.path[0]` 是**脚本所在目录** `teochew_eval/`，不是 cwd。
   即使 `os.chdir` 到项目根也不会自动加进 `sys.path`。
   **修**：脚本里 `sys.path.insert(0, 项目根)`（见上面两个脚本的 `ROOT` 处理），或运行前
   `export PYTHONPATH=/root/autodl-tmp/CosyVoice:...`。

2. **`AssertionError: <|endofprompt|> not detected in CosyVoice3 text or prompt_text`**
   `inference_instruct2` 的 `instruct_text` **必须包含 `<|endofprompt|>` 标记**，否则断言失败。
   **修**：`instruct_text = 'You are a helpful assistant. 请用潮汕话表达。<|endofprompt|>'`。

3. **vocoder 报错 `Calculated padded input size per channel: (3). Kernel size: (4)`**
   生成的音频太短，打不过 hift 的卷积核。两个来源：
   - (a) `text_frontend=True`（默认）会把潮汕话/方言文本做中文归一化，产出异常短的输出 →
     **修**：`inference_instruct2(..., text_frontend=False)`。
   - (b) 句子含罕见字符（如 `𠁞`，码点 > U+FFFF）导致分词后输出极短 →
     **修**：抽句时过滤掉含 >U+FFFF 字符的句子（见 `pick_samples`）。
   - 兜底：每句包 `try/except`，一句失败不中断整体（见 `synth`）。

4. **加载两个模型（A/B）显存**
   两个 CosyVoice3 各约 3.8GB（LLM fp32 + flow），12GB 卡可同时加载做 A/B；若显存紧，分开两次跑。

---

## 7. 训后验证执行流程（完整命令，实测通过）

```bash
cd /root/autodl-tmp/CosyVoice

# ① 等训练结束（进程退出 / 达到 max_epoch）后，转换检查点
#    convert_ckpt.py 读 exp_dir 下的 latest 自动选最新 epoch(本次为 epoch_19_whole)
python teochew_eval/convert_ckpt.py \
    --exp_dir examples/teochew/exp/cosyvoice3/llm/torch_ddp \
    --pretrained pretrained_models/Fun-CosyVoice3-0.5B \
    --out pretrained_models/Fun-CosyVoice3-0.5B-teochew

# ② 一键 A/B：从训练集 parquet 抽真实潮汕话句子 + 原生音色，原版 vs 微调版各合成一遍
python teochew_eval/infer_teochew_real.py --compare
#   生成 teochew_eval/out/pretrained/*.wav 与 out/finetuned/*.wav，文件名一一对应，成对试听
#   并写出 teochew_eval/out/{pretrained,finetuned}/manifest.txt 句子对照清单

# 只测微调版：
# python teochew_eval/infer_teochew_real.py \
#     --model_dir pretrained_models/Fun-CosyVoice3-0.5B-teochew
```

### 使用提示
- 想用**自己手写的句子**而非训练集句子，改用 `eval_teochew.py`（改 `TEochew_INSTRUCT`）。
- `--prompt` 默认是官方 `asset/zero_shot_prompt.wav`（通用音色）。`infer_teochew_real.py` 会自动
  从训练集抽一条原生 chao_shan_01 说话人音频作参考音色（`teochew_eval/prompt_chaoshan.wav`），
  方言内容由 instruct 决定、音色由这段参考音频决定，更贴近"潮汕音色"验证。
- 也可直接用官方 WebUI 试听：`python webui.py`（模型目录选 `...-teochew`）。

---

## 8. 路径速查

| 内容 | 路径 |
|---|---|
| 训练脚本 | `examples/teochew/run_train_12g.sh` |
| DeepSpeed 配置 | `examples/teochew/conf/ds_zero2_offload.json` |
| 训练日志 | `/root/autodl-tmp/CosyVoice/train_12g.log` |
| 训练检查点 | `examples/teochew/exp/cosyvoice3/llm/torch_ddp/epoch_N_whole/` |
| `latest` 文件 | `examples/teochew/exp/cosyvoice3/llm/torch_ddp/latest` |
| TensorBoard | `examples/teochew/tensorboard/cosyvoice3/llm/torch_ddp/` |
| 预训练模型 | `pretrained_models/Fun-CosyVoice3-0.5B/` |
| 检查点转换脚本 | `teochew_eval/convert_ckpt.py` |
| 验证脚本（手写句） | `teochew_eval/eval_teochew.py` |
| 验证脚本（训练集句，推荐） | `teochew_eval/infer_teochew_real.py` |
| 转换后的微调模型目录 | `pretrained_models/Fun-CosyVoice3-0.5B-teochew/`（执行转换后生成） |
| 合成输出 | `teochew_eval/out/{pretrained,finetuned}/*.wav` |
| 句子对照清单 | `teochew_eval/out/{pretrained,finetuned}/manifest.txt` |
| 参考音色（原生说话人） | `teochew_eval/prompt_chaoshan.wav`（infer_teochew_real.py 自动生成） |
| 参考音频（通用） | `asset/zero_shot_prompt.wav` |

---

## 9. 待办 / 备注

- [x] 训练 20 epoch（num_workers=2，ZeRO-2 offload）完成。
- [x] 检查点转换（epoch_19 → `...-teochew/llm.pt`）完成。
- [x] 推理 A/B 实测：原版 vs 微调版各 8 句；**主观确认微调版有明显潮汕话味道、非普通话发音**。
- [ ] 若想进一步增强：可加训练句数 / 调学习率 / 延长 epoch，再按第 7 节重跑转换+推理。
- [ ] 自动评测（WER / 相似度 / UTMOS）**本期不采用**，自动评测不可靠；如需后续补充再单独评估。
