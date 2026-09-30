# 潮汕话 CosyVoice3 微调操作文档

> 目标：用 teochew-data 的 ~1300 句潮汕话，对比「不微调（zero-shot）」vs「微调后」的效果。
> 框架：`feelins/CosyVoice` = Fun-CosyVoice 3.0（0.5B，Qwen2-based TTS）。
> 日期：2026-09-29

---

## 0. 结论先行（TL;DR）

| 问题 | 结论 |
|---|---|
| 数据能不能用 | 能。1309 句、单说话人 `chao_shan_01`、16kHz、约 2.31h、文本是潮汕话汉字 |
| 需要预提取 embedding/speech_token 吗 | **不需要**。CosyVoice3 支持训练时在线提取（`--onnx_path`） |
| 微调哪个模块 | **只微调 `llm`**（文本→语音 token 的关键），flow/hifigan 用 base 权重不动 |
| zero-shot 基线怎么打 | base 模型 + `请用闽南话表达`（闽南是最近的支持方言） |
| 微调后怎么触发 | 用自定义 instruct `请用潮汕话表达`（或复用 `请用闽南话表达`） |

---

## 1. 数据现状

`/root/autodl-tmp/teochew-data/`：

```
04_ChaoShan/train.list          1309 行，格式: wav路径|spk|lang|文本
wav/                            1331 个 wav（16kHz / 单声道）
```

`train.list` 每行 4 列（竖线 `|` 分隔）：

```
/root/autodl-tmp/GPT-SoVITS/data/04_ChaoShan/wav/00000001.wav|chao_shan_01|teochew|一念之慈，顶上生出灿烂圆光，者就是俺个不甚彻底个道德观念，
```

**关键点**：`train.list` 里的 wav 路径指向 GPT-SoVITS 的旧目录，实际音频在 `/root/autodl-tmp/teochew-data/wav/`，需重映射。

文本是潮汕话汉字（含 `呾`/`㩼`/`唔`/`俺`/`者` 等潮汕特有字），可直接进 Qwen tokenizer（这些字可能是低频/OOV，但 tokenizer 有字节级回退，不影响训练）。

---

## 2. 框架要点（CosyVoice3 训练数据流）

CosyVoice3 训练需要的是 **parquet 文件 + `data.list`**，不是原始 wav 列表。数据流：

```
wav.scp / text / utt2spk / instruct
        │
        ▼  (make_parquet_list.py 或自建脚本)
parquet (含 audio_data 字节 / text / spk / instruct)
        │
        ▼  (train.py 读取 data.list)
在线提取 embedding(campplus.onnx) + speech_token(speech_tokenizer_v3.onnx)
        │
        ▼
CosyVoice3LM 训练 (text_token + instruct_token → speech_token)
```

### 2.1 需要的字段（parquet 列）

| 列 | 说明 |
|---|---|
| `utt` | 语句 id |
| `audio_data` | wav 原始字节 |
| `wav` | wav 路径 |
| `text` | 潮汕话汉字文本 |
| `spk` | 说话人 id |
| `instruct` | 指令文本（方言控制用） |

### 2.2 在线 vs 离线提取

官方 `examples/libritts/cosyvoice3/run.sh` 注释明确写了：

> "NOTE embedding/token extraction is not necessary now as we support online feature extraction, but training speed will be influenced"

即：**CosyVoice3 支持在线提取，只需训练时传 `--onnx_path`，不必预提取 `utt2embedding.pt`/`utt2speech_token.pt`**。代价是训练时多花时间在线跑 onnx。

---

## 3. 第一步：数据准备

脚本已写好：`examples/teochew/local/prepare_data.py`。

### 3.1 运行命令

```bash
cd /root/autodl-tmp/CosyVoice/examples/teochew
/root/autodl-tmp/envs/cosyvoice/bin/python local/prepare_data.py \
    --src_list /root/autodl-tmp/teochew-data/04_ChaoShan/train.list \
    --wav_dir  /root/autodl-tmp/teochew-data/wav \
    --des_dir  data \
    --instruct "You are a helpful assistant. 请用潮汕话表达。<|endofprompt|>"
```

### 3.2 脚本做什么

1. 解析 `train.list`，把 wav 路径重映射到实际音频目录，剔除音频缺失/文本为空的条目。
2. 按 `seed=1986` 打乱后 **9:1 切分**（train ~1178 句 / cv ~131 句）。
3. 每个子集生成 `wav.scp`/`text`/`utt2spk`/`spk2utt`/`instruct` + `parquet/parquet_000000000.tar` + `parquet/data.list`。

### 3.3 产物

```
data/train/{wav.scp,text,utt2spk,spk2utt,instruct,parquet/{parquet_000000000.tar,data.list}}
data/cv/{...同上...}
```

> 说明：自建 parquet 时绕过了官方 `tools/make_parquet_list.py` 里 `spk_list` 未定义的小 bug（该 bug 只影响 `spk2parquet` 生成，不影响训练）。

### 3.4 instruct 的两个方案（重要，选一个）

| 方案 | instruct | 效果 |
|---|---|---|
| **A（对比最干净，推荐）** | `请用闽南话表达` | 复用基座已有的闽南 instruct，微调后"闽南"→潮汕口音；zero-shot 基线也用同一 instruct，只差权重 |
| **B（新增方言标签）** | `请用潮汕话表达` | 微调出独立标签，不动现有闽南；但 base 模型不认识该 instruct，zero-shot 基线得另用闽南 |

用方案 A 时，把上面命令的 `--instruct` 改成 `"You are a helpful assistant. 请用闽南话表达。<|endofprompt|>"`。

---

## 4. 第二步：Zero-shot 基线（不微调）

直接用 base 模型出结果，作为对照。

```python
import sys
sys.path.append('/root/autodl-tmp/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

cosyvoice = AutoModel(model_dir='/root/autodl-tmp/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
prompt_wav = '/root/autodl-tmp/teochew-data/wav/00000001.wav'  # 任意一句作参考

# 方式1：instruct 闽南（最接近潮汕的现成方言）
for i, j in enumerate(cosyvoice.inference_instruct2(
        '一念之慈，顶上生出灿烂圆光，者就是俺个不甚彻底个道德观念。',
        'You are a helpful assistant. 请用闽南话表达。<|endofprompt|>',
        prompt_wav, stream=False)):
    torchaudio.save('baseline_instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# 方式2：纯 zero-shot 克隆（克隆该潮汕音色，但口音仍是普通话）
for i, j in enumerate(cosyvoice.inference_zero_shot(
        '一念之慈，顶上生出灿烂圆光，者就是俺个不甚彻底个道德观念。',
        '一念之慈，顶上生出灿烂圆光，者就是俺个不甚彻底个道德观念。',
        prompt_wav, stream=False)):
    torchaudio.save('baseline_zeroshot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```

**测试集**：用 `data/cv` 里的文本（未参与训练的 ~131 句），批量生成基线音频。

---

## 5. 第三步：微调训练

### 5.1 准备训练配置

官方训练配置在 `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`（**不是**模型目录里的那个）。关键区别：训练配置的 `data_pipeline` 里多了 `compute_whisper_fbank`，用于在线提取 speech_token。

```bash
cd /root/autodl-tmp/CosyVoice/examples/teochew
mkdir -p conf
cp ../../libritts/cosyvoice3/conf/cosyvoice3.yaml conf/
cp ../../libritts/cosyvoice3/conf/ds_stage2.json conf/
```

**针对 1309 句小数据，建议改 conf/cosyvoice3.yaml 的 `train_conf`**：

```yaml
train_conf:
    optim: adam
    optim_conf:
        lr: 1e-5          # 保持 1e-5（小数据不宜更高）
    scheduler: constantlr
    max_epoch: 20         # 原 200 → 改小，防止过拟合
    grad_clip: 5
    accum_grad: 2
    log_interval: 100
    save_per_step: -1     # 每 epoch 存一次
```

### 5.2 合并 data.list

```bash
cd /root/autodl-tmp/CosyVoice/examples/teochew
cat data/train/parquet/data.list > data/train.data.list
cat data/cv/parquet/data.list   > data/dev.data.list
```

### 5.3 启动训练（单卡，只微调 llm）

```bash
cd /root/autodl-tmp/CosyVoice/examples/teochew
export CUDA_VISIBLE_DEVICES="0"
PRETRAIN=/root/autodl-tmp/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B

torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=1986 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    /root/autodl-tmp/CosyVoice/cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config conf/cosyvoice3.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --qwen_pretrain_path $PRETRAIN/CosyVoice-BlankEN \
    --onnx_path $PRETRAIN \
    --model llm \
    --checkpoint $PRETRAIN/llm.pt \
    --model_dir exp/cosyvoice3/llm/torch_ddp \
    --tensorboard_dir tensorboard/cosyvoice3/llm/torch_ddp \
    --ddp.dist_backend nccl \
    --num_workers 2 \
    --prefetch 100 \
    --pin_memory \
    --use_amp
```

**参数说明**：

| 参数 | 作用 |
|---|---|
| `--model llm` | 只训 LLM（文本→语音 token），方言适配的核心 |
| `--checkpoint .../llm.pt` | 从 base LLM 权重起训 |
| `--onnx_path $PRETRAIN` | 在线提取 embedding + speech_token |
| `--qwen_pretrain_path .../CosyVoice-BlankEN` | Qwen2 tokenizer + encoder |
| `--use_amp` | bf16 混合精度（省显存） |

训练产物在 `exp/cosyvoice3/llm/torch_ddp/`，每个 epoch 存 `epoch_{n}.pt` 和最终 `llm.pt`。

---

## 6. 第四步：微调后推理 + 对比

微调后，把新的 `llm.pt` 放进一个新的模型目录（保留 base 不动），flow/hifigan 复用 base：

```bash
# 复制 base 模型目录，替换 llm.pt
cp -r /root/autodl-tmp/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
      /root/autodl-tmp/CosyVoice/pretrained_models/Teochew-0.5B
cp exp/cosyvoice3/llm/torch_ddp/llm.pt \
      /root/autodl-tmp/CosyVoice/pretrained_models/Teochew-0.5B/llm.pt
```

然后跑和第二步**完全相同的推理脚本**（换 model_dir + 对应 instruct），得到微调后结果。

**对比维度**（用同一个测试集文本）：
1. 听感：口音是不是潮汕/闽南，还是普通话。
2. 内容一致性：有没有读错/漏读（可用 CER，对潮汕话需人工核对）。
3. 音色自然度。

---

## 7. 关键坑与注意事项

1. **数据量小 + 单说话人**：1309 句极易过拟合，`max_epoch` 一定改小（10~30），否则模型会"记住"这个说话人，丢掉 zero-shot 泛化（catastrophic forgetting）。
2. **instruct 必须和训练时一致**：训练用 `请用闽南话表达`，推理就得用 `请用闽南话表达`，否则触发不了。
3. **潮汕特有字是低频/OOV**：`呾`/`㩼`/`唔` 等字 Qwen tokenizer 可能按字节拆，模型学起来慢。若效果差，可考虑把文本规范化（潮汕字→普通话近义字），但会损失"潮汕味"，优先保持原字。
4. **别用模型目录里的 `cosyvoice3.yaml` 训练**：那个没有 `compute_whisper_fbank`，在线提取会缺 `whisper_feat`。要用 `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`。
5. **显存（实测结论）**：0.5B LLM 全量微调的 fp32 Adam 优化器状态（权重+梯度+m+v）约 8GB，加上激活 + onnx 语音 tokenizer + CUDA 上下文，**12GB 会 OOM（RTX 3080 Ti 实测）**。`max_frames_in_batch 2000→800`、`foreach: False` 都试过仍 OOM。**需 ≥24GB 显存**（RTX 3090/4090、A100 等），换机器后直接跑 `run_train.sh` 即可。
6. **flow/hifigan 不用训**：方言差异主要体现在 LLM 的 text→speech_token 映射，声学（mel/波形）是 speaker 无关的，复用 base 即可。
7. **`make_parquet_list.py` 的 `spk_list` bug**：官方脚本 `job()` 里 `spk_list` 未定义，会导致 `spk2parquet` 生成失败（但 parquet + data.list 正常）。本项目的 `prepare_data.py` 已自建 parquet，绕开此问题。
8. **缺 `pandas`**：训练管线 `parquet_opener`（`cosyvoice/dataset/processor.py`）里用 `df.to_pandas()`，所以 `pandas` 是训练硬依赖。本环境之前没装，需 `pip install pandas`（已补装 2.3.3）。
9. **deepspeed 版本必须 pin 0.15.1**：`train.py` 顶部无条件 `import deepspeed`，所以即使只用 `torch_ddp` 引擎也得装 deepspeed。本环境 torch=2.3.1，**不能装新版**（0.19.x 要求 torch≥2.4，导入会报 `torch.library.custom_op` 不存在）。`requirements.txt` pin 的是 `deepspeed==0.15.1`。

---

## 8. 附：文件清单

```
examples/teochew/
├── local/prepare_data.py       # 数据准备脚本（已写好，已执行）
├── conf/cosyvoice3.yaml        # 训练配置（从 libritts 复制，需改 max_epoch）
├── data/
│   ├── train/{wav.scp,text,utt2spk,instruct,parquet/parquet_000000000.tar,parquet/data.list}
│   └── cv/{...同上}
└── exp/cosyvoice3/llm/...      # 训练产物（待生成）
```

## 9. 执行进度与剩余步骤（checklist）

### 已完成 ✅

- [x] 数据准备：`prepare_data.py` 已执行，**1309 句全部有效（0 缺失）**，切分 `train 1179 / cv 130`，parquet + `data.list` 生成并验证通过（列结构、音频加载、采样率 16kHz 均正常）。
- [x] 环境依赖：`pandas` 已补装（2.3.3）。
- [x] 文档：`usage_analysis.md`（四个问题分析）、本文件（微调操作）。

### 待完成 ⬜

- [ ] 降级 deepspeed 到 0.15.1：
  ```bash
  /root/autodl-tmp/envs/cosyvoice/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple "deepspeed==0.15.1"
  ```
- [ ] 复制训练配置 + 改 `max_epoch`（200 → 20）：
  ```bash
  cd /root/autodl-tmp/CosyVoice/examples/teochew
  mkdir -p conf
  cp ../../libritts/cosyvoice3/conf/cosyvoice3.yaml conf/
  cp ../../libritts/cosyvoice3/conf/ds_stage2.json conf/
  # 手动把 conf/cosyvoice3.yaml 里 train_conf.max_epoch 改为 20
  ```
- [ ] 合并 data.list：
  ```bash
  cat data/train/parquet/data.list > data/train.data.list
  cat data/cv/parquet/data.list   > data/dev.data.list
  ```
- [ ] 打 zero-shot 基线（§4，不微调）。
- [ ] 启动微调训练（§5.3，`--model llm` 单卡）。
- [ ] 微调后推理 + 对比（§6）。
