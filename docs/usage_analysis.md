# CosyVoice（Fun-CosyVoice3-0.5B）使用分析

> 针对四个问题的代码级核实结论。环境已就绪（`cosyvoice` 环境 + `pretrained_models/Fun-CosyVoice3-0.5B` 模型）。
> 日期：2026-09-29

---

## 0. 背景：这个模型到底是什么

`feelins/CosyVoice` 是 **Fun-CosyVoice 3.0**（0.5B 参数），一个基于 Qwen2 LLM 的 TTS：

- 文本侧：`Qwen2` 编码器（`CosyVoice-BlankEN` tokenizer，151K 词表）读文本 → 生成离散**语音 token**（6561 码表，25 token/秒）。
- 声学侧：flow-matching DiT + HiFT-GAN 把语音 token 还原成波形。
- 语音 tokenizer（`speech_tokenizer_v3.onnx`）用 Whisper 那套多语标签做「语音↔token」。

README 宣称支持：**9 种语言**（中/英/日/韩/德/西/法/意/俄）+ **18+ 种汉语方言/口音**（广东、闽南、四川、东北…）+ 多语/跨语 zero-shot 克隆 + instruct（方言/情绪/语速）。

---

## 1. webui.py 能不能直接启动？

**结论：不能直接用，需要装依赖 + 改一处代码。**

### 1.1 现状

- `webui.py` 是 Gradio 界面，4 种模式：
  - `预训练音色`（SFT，需 CosyVoice-300M-SFT 模型，本模型无内置音色）
  - `3s极速复刻`（zero-shot，可用）
  - `跨语种复刻`（cross-lingual，可用）
  - `自然语言控制`（instruct，**不可用，见下**）
- 默认 `--model_dir` 是 `CosyVoice2-0.5B`，跑 Fun-CosyVoice3 要改成：
  ```bash
  python webui.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
  ```

### 1.2 缺依赖

`gradio`、`fastapi` 当前**未安装**（当初装依赖时跳过了 webui 相关）。启动前需补：

```bash
/root/autodl-tmp/envs/cosyvoice/bin/pip install -i https://mirrors.cloud.tencent.com/pypi/simple gradio fastapi
```

### 1.3 关键坑：方言 instruct 模式用不了

`webui.py` 第 114 行「自然语言控制」调用的是：

```python
cosyvoice.inference_instruct(...)   # 老接口
```

而 `inference_instruct` 在源码里有断言 `self.__class__.__name__ == 'CosyVoice'`（仅 CosyVoice1 / 300M-Instruct 可用）。Fun-CosyVoice3 用的是 **`inference_instruct2`**（带 `<|endofprompt|>` 指令的版本）。

**所以 webui 里的方言控制默认是坏的**，要把第 114 行改成 `inference_instruct2`（并相应调整参数），才能用「请用闽南话表达」这类方言指令。

### 1.4 小结

| 项目 | 状态 |
|---|---|
| 启动 webui | 需装 gradio/fastapi |
| zero-shot / cross-lingual | 直接可用 |
| 方言 instruct | **需改 webui.py**（换 `inference_instruct2`） |

---

## 2. 潮汕/闽南话：zero-shot 就能用，还是要微调？

**结论：已支持的方言（含闽南）zero-shot + instruct 即可用，无需微调。潮汕属闽南，方向正确。**

### 2.1 证据

- `cosyvoice/utils/common.py` 的 `instruct_list` 明确含 **17 种方言**，其中就有 `"请用闽南话表达"`。
- `cosyvoice/tokenizer/tokenizer.py` 的 `LANGUAGES` 含 `"minnan": "minnan"`。
- 实测：`inference_instruct2('落雨天时…', '请用闽南话表达', prompt_wav)` 成功合成，RTF≈0.6。

### 2.2 潮汕 vs 闽南的差异

- 潮汕话（潮州/汕头）是**闽南语系**的次方言，写的是**汉字**，所以文本侧完全无障碍（汉字进 Qwen tokenizer 是正常 1 字 1 token）。
- 但 `请用闽南话表达` 给的是「泛闽南」口音（更接近闽南核心区，如厦门/台湾），**不保证精确到潮汕特有的声调、韵尾**。

### 2.3 结论

- **直接可用**：用 `inference_instruct2` + `请用闽南话表达`，零训练成本。
- **若要精确潮汕口音**：才需要微调（给模型喂潮汕「音频+汉字」语料，用 `teochew-data` 的 1309 条即可起步）。

---

## 3. 藏语→英语音标 / 汉语拼音映射：能不能跑通？

**结论：这是最有希望、且零训练成本的路径——强烈建议先试。因为 tokenizer 里已经内置了这两套音素符号。**

### 3.1 关键发现：模型原生支持音素符号

`CosyVoice3Tokenizer` 的 special tokens 里，**已经注册了两套音素**：

- **ARPABET（英语词典音标）**：`[AA] [AE] [AH] [AO] [B] [CH] [D] [DH] [EH] [ER] [F] [G] [HH] [IH] [IY] [JH] [K] [L] [M] [N] [NG] [OW] [OY] [P] [R] [S] [SH] [T] [TH] [UH] [UW] [V] [W] [Y] [Z] [ZH]`（带 0/1/2 重音）
- **汉语拼音（带声调）**：`[a] [ai] [an] [ang] [ao] [b] [c] [ch] [d] [e] [ei] [en] [eng] [f] [g] [h] [i] [ian] [in] [ing] [iu] [j] [k] [l] [m] [n] [o] [ong] [ou] [p] [q] [r] [s] [sh] [t] [u] [un] [uo] [v] [w] [x] [y] [z] [zh]` + 全套声调变体（`[ā][á][ǎ][à]` 等）。

### 3.2 证据：模型支持「读音热修」（hotfix）

`example.py` 里有官方示例：

```python
cosyvoice.inference_zero_shot('高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。', ...)
```

`[j][ǐ]` 就是用拼音 token 就地纠正「给」的读音。说明**模型被训成能读懂方括号里的音素符号并按其发音**。

### 3.3 为什么藏语之前跑不通

- 藏文是**独立文字系统**（藏文 Unicode U+0F00–U+0FFF），Qwen tokenizer 对它是字节级回退（93 个 token 全是乱码），LLM 没学过、生成几乎为空。
- 即：问题不在「藏语发音难」，而在「藏文文字进不去」。

### 3.4 映射方案的思路（零训练）

把藏语发音转成模型认识的符号，两条路：

1. **转汉语拼音（首选，因为有声调）**：
   - 藏语音节 → 最接近的拼音声母+韵母+声调。
   - 例：`ཀ` [ka] → `[g][ā]`（藏语 ka 不送气，近似 g）；`ཁ` [kʰa] → `[k][ā]`（送气，近似 k）。
   - 藏语声调（高/低）用拼音声调符号近似。
2. **转 ARPABET（英语音标）**：
   - 藏语塞音送气/不送气、擦音等用 ARPABET 近似（`[P]`/`[B]`、`[T]`/`[D]`、`[K]`/`[G]`、`[S]`/`[Z]`、`[SH]`、`[ZH]`、`[NG]`…）。

### 3.5 局限（需实验验证）

- **音素覆盖**：藏语有卷舌音（ཏྲ/དྲ）、送气对立、鼻化等，拼音/ARPABET 不能 100% 精确，只能近似。
- **纯音素长句**：模型被训成「中文/英文句子 + 局部音标热修」的模式，整句都是音素符号是否稳定、自然，需要实测。
- **实现**：需要一个「藏文→音素符号」的转换脚本（类似之前给 DiaMoE 写的 `prepare_tibetan.py`，只是目标从 IPA 换成拼音/ARPABET）。

### 3.6 结论

**强烈建议先做这个实验**：写个藏文→拼音符号的转换，直接喂 `inference_zero_shot`，成本极低（几小时），若成功就绕过了「藏文进不去」的墙。

---

## 4. 吴语（温州）这种偏僻方言

**结论：没有现成支持，是最贵的一条路。**

### 4.1 现状

- `LANGUAGES` 里有 `"wuyu": "wuyu"`（吴语标签），但这是**语音 tokenizer（ASR 侧）**的标签，不代表文本 LLM 能合成吴语。
- `instruct_list`（文本合成侧的方言清单）里**没有「吴语/温州」**。最接近的是 `"请用上海话表达"`（上海=北部吴语）。
- 温州是**南部吴语**，和上海话（北部吴语）差异很大（声调系统、连读变调、词汇都不同）。

### 4.2 能试的近似手段

| 手段 | 可行性 |
|---|---|
| `请用上海话表达` | 能出「吴语系」音色，但不是温州话 |
| 跨语种 `<|wuyu|>` 标签 | 不确定 LLM 是否训过，需实测 |
| zero-shot 用温州参考音频 | 只能克隆音色，口音仍受指令控制 |

### 4.3 结论

- **不能直接支持**温州话。
- 想精确支持，只能**微调**（喂温州「音频+文本」语料）。
- 但「温州话」与「潮汕话」在框架里是同一类问题——**都是汉字书写、方言发音**，所以微调路径完全可复用（用 `teochew-data` 那套流程）。

---

## 5. 总结与建议（按性价比排序）

| 优先级 | 事项 | 成本 | 预期 |
|---|---|---|---|
| **① 立刻做** | 潮汕/闽南：`inference_instruct2` + `请用闽南话表达`，批量跑 `teochew-data` 出样例 | 零训练 | 高，方向已验通 |
| **② 值得试** | 藏语→拼音/ARPABET 音素映射（利用内置音素 token + hotfix 机制） | 几小时脚本 | 中高，可能打通藏语 |
| **③ 改 webui** | 装 gradio/fastapi + 把 instruct 换成 `inference_instruct2` | 半小时 | 中（图形界面，方便你自测） |
| **④ 吴语/温州** | 需微调，复用潮汕流程 | 高 | 中（看语料量） |

### 一句话结论

- **潮汕话**：直接用，不用训练。
- **藏语**：换成「音素符号映射」这条路子，很可能跑通（模型原生支持拼音/音标符号）。
- **吴语/温州**：没有现成支持，最贵，放最后。
- **webui**：要改代码才能用方言功能，否则只能当 zero-shot 用。
