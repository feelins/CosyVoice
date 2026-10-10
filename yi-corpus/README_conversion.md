# 彝语 → CosyVoice3 微调：IPA / X-SAMPA 处理方案与中间结果

本目录包含把 `yi-corpus` 彝语语料转成 CosyVoice3 可训练表示（X-SAMPA）的脚本、
中间结果与训练准备文件。**未启动训练**（按需求先只做方案/脚本/中间结果）。

## 1. 数据现状（实测）

- `metadata.csv`（UTF-8 BOM，用 `utf-8-sig` 读取）：1766 行，`id / source / wav /
  duration / text / text_zh / speaker_note`。
- `metadata.jsonl`：1766 行，额外有 `text_ipa / text_roman / text_en / license`。
- `wavs/`：1766 个 `.wav`，16kHz / 单声道 / 16bit，`duration` 字段与音频精确一致，
  无缺失、无多余。
- **两个来源（方言不同）**：
  - `04-iiiwav`（329 句，0.19h）：凉山诺苏语，圣经朗读切分。`text` = 凉山拼音
    （如 `yi su cyt`），带调字母 `t/x/p` 在音节末尾，无标记=中平调。**无声调字段**。
  - `06-chuxiong-yi-meige`（1437 句，5.43h）：楚雄彝语（梅葛史诗）。`text` = **已转写
    的 IPA**，调值用上标数字对（`li³³ ŋo³³ li⁵⁵ pe³³ pe²² næ³¹ ...`，即 55/33/31/22/34）。
  - 时长：06 有 21 句 >30s（最长 287s）、11 句 <1s；04 有 67 句 <1s。这些极端值在
    训练配置 `filter`（1s–60s）里自动丢弃。

## 2. 为什么用 X-SAMPA（而不是原生 IPA）

与藏语用 Wylie 罗马化的思路一致：原生彝语 IPA 大量使用 `IPA Extensions` 块、上标数字
（`⁰¹²³⁴⁵`）、`ʦ ʨ ɿ ʑ ŋ æ ə ɯ ɔ` 等字符，Qwen 底座分词器覆盖很差，会被切成乱码/
未知符，训练学不到稳定音素边界。X-SAMPA 是纯 ASCII 音标，Qwen 词表 100% 覆盖，且
CosyVoice 训练时 `text` 字段是**直接由 Qwen 分词**（不做 g2p），所以喂 X-SAMPA 即可。
推理时也必须 `text_frontend=False` + instruct「请用彝语表达」。

## 3. 转换管线

```
06 IPA 文本  ──ipa2xsampa──► X-SAMPA   (主路径，准确)
04 凉山拼音  ──roman2ipa──► IPA ──ipa2xsampa──► X-SAMPA
```

- `scripts/ipa2xsampa.py`：彝语微调版 IPA→X-SAMPA。新增上标数字→ASCII 数字、
  `ʦ→ts ʨ→tc ʥ→dj ʣ→dz ʧ→tS ɿ→z ȵ→J ȶ→t ȡ→d ȴ→l`、`'→h`、`I→i`、`，→,`，
  其余 ASCII 原样保留。`--check` 可打印未覆盖符号。
- `scripts/roman2ipa.py`：凉山拼音→IPA（声母/韵母/声调按标准凉山方案；声调写成上标
  数字对 55/34/21/33）。解析失败的 token 原样保留。
- `scripts/build_dataset.py`：读 `metadata.csv` + `metadata.jsonl`，按来源转换，写出
  中间结果（见第 4 节）与切分。
- `scripts/prepare_parquet.py`：把切分 tsv 转成 CosyVoice3 训练 parquet（列
  `utt/audio_data/wav/text/spk/instruct`，`text`=X-SAMPA，`instruct` 固定）。

## 4. 中间结果（已生成，在 `processed/`）

- `all_xsampa.tsv` / `all_xsampa.jsonl`：全量 1766 句，含 `text_xsampa / text_ipa /
  text_roman / use_for_train / duration / source`。
- `stats.json`：来源计数、时长分布、切分规模、空值统计。
- `splits/train.tsv, valid.tsv, test.tsv`：06+04 按来源分层切分。
- `splits/train_06.tsv, valid_06.tsv, test_06.tsv`：**仅 06（楚雄）**，推荐首训，
  避免方言/声调系统混杂。
- `data/{train,cv,test}/parquet/...` + `data/train.data.list` + `data/dev.data.list`：
  由 `prepare_parquet.py` 生成的训练 parquet（已生成，方便直接开训）。

切分 tsv 格式：`idx\t<wav相对路径>\t<xsampa>`。

## 5. 训练（未执行）

配置/启动脚本已就位，但**未启动**：

- `conf/cosyvoice3.yaml`：复制自 `examples/tibetan`，仅改了模型/日志目录为 `yi`；
  只微调 `llm`（flow/hift 冻结），`filter` 1s–60s，`text` 即 X-SAMPA。
- `conf/ds_zero2_offload.json`：12GB 单卡 ZeRO-2 offload。
- `run_train_12g.sh`：12GB 单卡启动器（改了 `yi` 目录）。

开训前先确认：

```
cd /root/autodl-tmp/CosyVoice/yi-corpus
bash run_train_12g.sh
```

推理（训练后）须：

```python
# text_frontend=False，输入 X-SAMPA，instruct 固定「请用彝语表达」
# 例：text = "ni33 No33 li55 pe33 pe22 n{31 g@31 nja33 le33 No33"
```

## 6. 已知问题与下一步

1. **方言/声调混杂**：04（凉山 55/34/21/33）与 06（楚雄 55/33/31/22/34）声调系统不同，
   数字调值不可直接比较。建议首训只用 `splits/*_06.tsv`（已生成）。
2. **04 声调/清化音近似**：`hm/hn` 暂按 `m/n`、`hl→ɬ`、`nz/nr/nj` 前鼻化塞音按近似；
   如需精确，请母语者校验 `roman2ipa.py` 的 `INITIAL_IPA`。
3. **多说话人**：06 含多名演唱者（郭自安、罗荣秀等），当前 `spk` 统一填 `'yi'`
   （CosyVoice3 用音频在线提取说话人嵌入，`use_spk_embedding=False`），如需多说话人
   克隆可在 `prepare_parquet.py` 按 `speaker_note` 派生 `spk`。
4. **训练规模偏小**：约 5.6h（06 约 5.4h）。建议先跑 `*_06` 验证可行性与音色，再决定是否
   并入 04 或扩充数据。
