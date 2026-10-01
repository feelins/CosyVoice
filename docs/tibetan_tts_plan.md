# 藏语 TTS 方案分析（基于现有底座）

> 目标：在现有底座模型上实现可用的藏语（བོད་སྐད་ / Bod skad）TTS。
> 背景：潮汕话 CosyVoice3 微调已验证有效（见 `teochew_cosyvoice3_finetune_eval.md`）。
> 本文盘点磁盘上已有的藏语资产，分析两条底座路线并给出可执行方案。
> 时间：2026-10-01。

---

## 0. 结论（TL;DR）

- **主推底座：DiaMoE-TTS**（F5-TTS + 统一 IPA 方言前端 + MoE + LoRA/Conditioning Adapter）。
  磁盘上已有完整藏语管线脚本、配置、IPA 音素草稿与检查点，**是风险最低、最快可落地的路径**。
- **备选底座：CosyVoice3 0.5B**（即潮汕话同款管线）。可行，但底座 LLM 分词器**不含藏文 Unicode（覆盖 0/13）**，
  必须用 **Wylie 罗马化**作训练文本，且声学 tokenizer 对藏语音系覆盖未知 —— 仅在需要 CosyVoice3 生态时考虑。
- 现有藏语语料（约 4.69h、16kHz）已齐备，数据不是瓶颈。

---

## 1. 现有藏语资产盘点（均在 `/root/autodl-tmp`）

| 资产 | 路径 | 说明 |
|---|---|---|
| 语音+文本语料 | `tibetan_wavs/wav/`（5923 个 wav，16kHz） | 实际音频 |
| 藏文文本 | `train-uni.tsv` / `valid-uni.tsv` / `test-uni.tsv`（各 4703/589/589 行） | 字段 `path \t sentence`（藏文原形） |
| Wylie 罗马化 | `train-wylie.tsv` / `valid-wylie.tsv` / `test-wylie.tsv` | 同上的 Wylie 转写（ASCII，分词器友好） |
| IPA 音素草稿 | `tibetan_wavs/phoneme_draft.tsv`（1917 行） | 字段 `unit/count/structure/ipa/notes`，已把藏语音节映射到 IPA（如 `pa→pa55`、`dang→tʰã13`） |
| **DiaMoE 藏语管线** | `DiaMoE-TTS/` | 见下 |
| 藏语声调识别项目 | `train_v1/` | 是 ToBI 风格**声调/韵律识别**（MMS+CTC），非 TTS，仅作资产参考 |

### DiaMoE-TTS 藏语管线（已就绪，几乎"一键"）
- `DiaMoE-TTS/run_tibetan.sh`：`manifest → arrow → check → train` 四步；预期 train 4387 / valid 553 / test 549，OOV=0。
- `DiaMoE-TTS/reset_tibetan_female.sh`：重置为**女声单说话人**训练（解决之前男女混训音色漂移）。
- `DiaMoE-TTS/diamoe_tts/src/f5_tts/train/datasets/prepare_tibetan.py` + `prepare_tibetan_arrow.py`：
  读 `tibetan_wavs` + tsv + `phoneme_draft.tsv`，生成清单/arrow。
- 配置：`diamoetts_tibetan_peft.yaml`（PEFT/LoRA，轻量）、`diamoetts_tibetan_fullft.yaml`（全量）、`diamoetts_tibetan_moeft.yaml`（MoE 微调）；甚至还有 `diamoetts_teochew_peft.yaml`。
- 检查点：`checkpoints/best_bo_fullft_7000.pt`、`10ep_mlpEXP_9_*.lora.pt`（bo=藏语）等。
- 环境：`/root/autodl-tmp/envs/diamoetts`（需先 `pip install -e .`）。

---

## 2. 方案 A（主推）：DiaMoE-TTS + 藏语

### 2.1 为什么选它
- **IPA 统一表示**：藏语作为"方言"接入，直接用 `phoneme_draft.tsv` 的 IPA，无需任何 tokenizer 适配 ——
  绕开了 CosyVoice3 的藏文分词死角。
- 已脚本化、已配置、已有检查点；F5-TTS + PEFT/LoRA 在 12GB 单卡可跑。
- 方言前端（`dialect_frontend/`）本身支持 IPA，跨方言一致。

### 2.2 执行步骤（直接复刻现有脚本）
```bash
cd /root/autodl-tmp/DiaMoE-TTS

# ① 数据准备（manifest + arrow + 校验），预期 train 4387/valid 553/test 549, OOV=0
bash run_tibetan.sh all

# ②（可选）先只训女声，避免男女混训音色漂移
bash reset_tibetan_female.sh          # 数据就绪；末尾加 train 可直接后台启动训练

# ③ 训练（PEFT，轻量）
bash run_tibetan.sh train

# 监控
tail -f /root/autodl-tmp/tmp/tibetan_train.log
tensorboard --logdir DiaMoE-TTS/diamoe_tts/ckpts
```
- 显存不足时降 `batch_size_per_gpu`（800→600→400）与 `max_samples`（16→12）。
- `reset_tibetan_female.sh` 默认只训 chapter 01（女声），解决之前音色漂移问题。

### 2.3 验证
- 用 `DiaMoE-TTS` 的 gradio / `batch_infer.sh` 合成样例，主观试听藏语自然度、音色一致性。
- 对比 `best_bo_fullft_7000.pt`（全量）与 LoRA 检查点，选更优者。

---

## 3. 方案 B（备选）：CosyVoice3 0.5B + 藏语（Wylie）

### 3.1 可行性证据（实测底座分词器）
- vocab 大小 151643；**藏文 Unicode 字符覆盖 0/13**（完全不在词表，会被拆成无意义字节 token）；
  **Wylie 罗马化覆盖 12/13**，分词干净（`['b','od','Ġsk','ad',...]`）。
- 结论：**必须用 Wylie 列作 `text`**（不能用藏文原形），否则序列爆炸且学不到语义。
- `text_frontend=False` 必须（中文前端处理不了藏语/Wylie）。
- instruct 同潮汕话：`"请用藏语表达。<|endofprompt|>"`（可加方言标注，如"卫藏/康/安多"）。
- 风险：CosyVoice3 声学 tokenizer（`speech_tokenizer_v3`）对藏语音系覆盖未知，需先抽样合成验证；
  若 token 覆盖差，音质会受限。

### 3.2 数据改造
- 复用 `*-wylie.tsv` 的 `sentence` 列作 `text`，构建 CosyVoice 的 parquet（字段 `text/instruct/audio_data`），
  方法同潮汕话 `examples/teochew/data` 的构造（需自己写藏语版构建脚本，仓库暂无现成）。
- 参考音色用 `tibetan_wavs` 中某条原生藏语人声。

### 3.3 训练/推理
- 训练：直接复用潮汕话 `examples/teochew/run_train_12g.sh`（改数据/模型目录即可，DeepSpeed ZeRO-2 offload 方案通用）。
- 推理：复用 `teochew_eval/infer_teochew_real.py` 思路（改为读藏语 parquet + 原生音色 + Wylie 文本）。

---

## 4. 两个底座怎么选

- 目标只是"出一个能用的藏语 TTS" → **选 A（DiaMoE）**：已有完整管线，风险最低、最快。
- 目标要"CosyVoice3 生态里的藏语方言" → 选 B，接受 Wylie + 声学 tokenizer 限制。
- 也可并行：A 出可用模型，B 验证 CosyVoice3 跨方言泛化能力。

---

## 5. 数据盘清理（与藏语无关，但连带处理）

潮汕话相关占用约 145G：`examples/teochew/exp` 135G（训练检查点）+ `pretrained_models/Fun-CosyVoice3-0.5B` 9.1G（底座）
+ `/root/autodl-tmp/teochew-data` 0.5G（重复数据）。

- **保留**：`pretrained_models/Fun-CosyVoice3-0.5B-teochew`（最终模型，含 `llm.pt`）与其依赖的底座。
- **可删**：`examples/teochew/exp`（检查点，转换后已无用）、`examples/teochew/data`（训练 parquet，不再扩数据）、`/root/autodl-tmp/teochew-data`。
- 删除后可释放约 135G，磁盘从 198G/200G 降到约 63G 占用。藏语资产（`train_v1`/`DiaMoE-TTS`/`tibetan_wavs`/tsv）全部保留。

---

## 6. 待办

- [ ] 跑 `bash run_tibetan.sh all` 确认数据 OOV=0、数据集可加载。
- [ ] 选 PEFT 还是 fullft；建议先 PEFT 出基线。
- [ ] 处理女声/男声混训音色漂移（用 `reset_tibetan_female.sh`）。
- [ ] 主观试听评估，必要时加数据 / 调 config。
- [ ] （备选）若走 CosyVoice3：先写藏语 parquet 构建脚本 + 用 Wylie 文本，抽样验证 `speech_tokenizer_v3` 覆盖。
