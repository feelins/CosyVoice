# 藏语 TTS 方案（CosyVoice3 0.5B）

> 目标：在 **CosyVoice3 0.5B** 底座上实现藏语 TTS，复用潮汕话已验证的微调 + 零样本复刻管线。
> 更新：2026-10-01。
> 状态：DiaMoE-TTS 已实测放弃（见 §0），本方案改为**纯 CosyVoice3** 路线。

---

## 0. 为什么放弃 DiaMoE-TTS

- 已在**藏语**、**潮汕话**上多次尝试 DiaMoE-TTS（F5-TTS + 统一 IPA 方言前端 + MoE + LoRA/Conditioning Adapter）。
- 现象：**IPA 发音非常好，但合成音色与参考音频完全不一致**（speaker adaptation / voice cloning 失败）。
- 结论：DiaMoE 的"方言适配"在**音色保持**上不稳定，放弃。
- 改用 CosyVoice3 的 **speaker-conditioned 零样本复刻**方案——这正是潮汕话已验证有效的点（finetuned 模型音色/方言都贴近参考人声）。

---

## 1. 可行性证据（底座分词器实测）

底座 LLM 分词器 `pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN`（vocab=151643）：

- **藏文 Unicode 字符覆盖 0/13** → 藏文原形会被拆成无意义字节 token，**不可用**。
- **Wylie 罗马化覆盖 12/13**，分词干净：`['b','od','Ġsk','ad',...]`。
- 结论：藏语训练/推理文本**必须用 Wylie 罗马化**（来自 `*-wylie.tsv` 的 `sentence` 列），**不能用藏文原形**。

---

## 2. 复用潮汕话已验证管线（`/root/autodl-tmp/CosyVoice`）

| 复用资产 | 用途 |
|---|---|
| `examples/teochew/run_train_12g.sh` | DeepSpeed ZeRO-2 CPU offload，12GB 单卡训练 |
| `examples/teochew/conf/ds_zero2_offload.json` | 同上配置 |
| `examples/teochew/cosyvoice.yaml`（训练配置） | 复用模型/数据配置 |
| `teochew_eval/convert_ckpt.py` | 把训练检查点转成推理用 `llm.pt` |
| `teochew_eval/infer_teochew_real.py` | 零样本复刻参考音色的推理（读 parquet + prompt） |
| `pretrained_models/Fun-CosyVoice3-0.5B` | 底座 |

> 注：Teochew 训练 parquet 已被清理，但其**构建脚本**需复用——先定位（`examples/teochew` 下构建 parquet 的脚本），再复制改写成藏语版。

---

## 3. 数据准备（新建 `examples/tibetan`）

语料已在盘：

- 音频：`tibetan_wavs/wav/`（5923 个 wav，16kHz）
- 文本：`train/valid/test-wylie.tsv`（Wylie 罗马化，`path \t sentence`），规模 train 4387 / valid 553 / test 549

需要写一个构建脚本（仿 Teochew 的 parquet 构造），产出 CosyVoice 训练 parquet：

- 字段：`text`（**Wylie**）、`instruct`（`"请用藏语表达。"`，可加方言标注如"卫藏/康/安多"）、`audio_data`（wav bytes）。
- 音频路径：tsv 的 `path` → `tibetan_wavs/wav/<path>`。
- 采样率：复用 Teochew 数据构建里的重采样逻辑，统一到 CosyVoice3 底座要求（与其一致即可，Teochew 同为 16k 源且已跑通）。
- 输出：`examples/tibetan/data/train/parquet/...`、`valid`、`test` + `train.data.list`、`valid.data.list`。

快速做法：复制 Teochew 数据构建脚本 → 改名藏语版 → 文本列换成 Wylie、音频根目录换成 `tibetan_wavs/wav`。

---

## 4. 训练

```bash
cd /root/autodl-tmp/CosyVoice
cp -r examples/teochew examples/tibetan          # 再改数据/模型路径

# run_train_12g.sh 关键参数（与潮汕话一致）
#   --train_data_set_list  examples/tibetan/data/train.data.list
#   --cv_data_set_list     examples/tibetan/data/valid.data.list
#   --model_dir            examples/tibetan/exp/cosyvoice3/llm/torch_ddp
#   --train_conf           examples/teochew/cosyvoice.yaml   (复用)
#   text_frontend=False   （中文前端处理不了 Wylie）
bash examples/tibetan/run_train_12g.sh
```

显存不足时沿用 ZeRO-2 offload（已验证 12GB 可跑）。

---

## 5. 转模型 + 零样本复刻推理

```bash
# 转换检查点为 llm.pt（改 convert_ckpt.py 路径指向 examples/tibetan/exp/.../latest）
python teochew_eval/convert_ckpt.py
# 用一条原生藏语人声作 prompt，零样本复刻音色（改 infer_teochew_real.py：读藏语 parquet + Wylie 文本 + 藏语 prompt wav）
python teochew_eval/infer_teochew_real.py
```

输出目录：`pretrained_models/Fun-CosyVoice3-0.5B-tibetan/`。

---

## 6. 风险与验证（决定成败的关键）

1. **主要风险：声学 tokenizer（`speech_tokenizer_v3`）对藏语音系覆盖未知**。
   验证：取几条藏语 wav 跑 `speech_tokenizer_v3.onnx`，看 token 是否集中在合法范围、是否大量异常。
   影响：CosyVoice 用 flow-matching 生成，且目标是**音色复刻**（DiaMoE 的失败点），token 覆盖差主要影响自然度而非音色一致性。
2. Wylie 是罗马化而非音素，CosyVoice 需学会 Wylie→藏语音系映射；潮汕话（同样非标准文字）已证明可行。
3. 多说话人：语料含女声（OTR002-01）+ 男声（OTR002-02~09），用参考音色做零样本复刻即可，无需合并训练。

---

## 7. 评估口径（针对 DiaMoE 的失败点）

- **音色一致性**：合成音 vs 参考音，主观 + 客观（speaker embedding 余弦相似度）。
- **发音/方言正确**：藏语母语者试听（Wylie 文本应被念成正确藏语，而非按英文读）。
- **自然度**：MOS。
- 对照：base（无微调）vs finetuned（用 instruct "请用藏语表达"）。

---

## 8. 待办

- [ ] 定位并复用 Teochew 的 parquet 构建脚本，写出藏语版（Wylie + 重采样）。
- [ ] 抽样验证 `speech_tokenizer_v3` 对藏语的 token 覆盖。
- [ ] 训练（先小步数看 loss 下降）→ 转 `llm.pt`。
- [ ] 零样本复刻参考音色推理，重点核对"发音 + 音色"是否都接近（DiaMoE 失败之处）。
- [ ] 必要时加 instruct 方言标注 / 调数据配比 / 试不同参考说话人。

---

## 附：数据盘清理（已完成）

潮汕话相关约 145G：`examples/teochew/exp`（检查点，已删）、`examples/teochew/data`（训练 parquet，已删）、
`/root/autodl-tmp/teochew-data`（重复数据，已删）。

系统盘（overlay 30G）曾 100% 满，因 `/root/miniconda3`(13G) 与 `/root/.cache`(4.9G) 在系统盘；
已整体转移到数据盘（`/root/autodl-tmp/miniconda3`、`.cache`）并软链，系统盘降至 63%（12G 可用）。
