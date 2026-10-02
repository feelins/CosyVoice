# CosyVoice3 跨语种扩展：文本表示方案分析（拉丁罗马化语言 / IPA 语言如彝语）

> 配套文档：`docs/tibetan_cosyvoice3_finetune.md`（藏语 Wylie 落地实验）。
> 本文回答两个问题：
> 1. 能用拉丁/ASCII 罗马化的语言，是否都能用同一套微调流程训练？
> 2. 只能用 IPA 表示的低资源语言（如彝语），IPA 如何映射进 CosyVoice3 框架？
>
> 核心前提（来自藏语实验）：CosyVoice3 的 LLM 文本通道 = **Qwen2.5 BPE 分词器**
> （`text_frontend=False` 时直接对输入文本做 subword 分词，不经过独立音素词表）。
> 因此“一个语言能不能训”等价于“它的文本字符能否在 `vocab.json` 里切成干净 token”。

---

## 0. 核心结论（TL;DR）

| 语言类型 | 能否直接训 | 方案 |
|---|---|---|
| 拉丁/ASCII 可罗马化的语言（藏语 Wylie、日语 romaji、越南语、汉语拼音、印尼语…） | ✅ 直接可训 | 把 `text` 设为罗马化串，零代码改动 |
| 带变音符的拉丁语（越南语、非洲声调语） | ✅ 多数可训 | 先跑覆盖率检查；变音符不在词表则去变音符或自定义 ASCII |
| **IPA 表示的低资源语言（彝语等）** | ⚠️ 不能直接喂 IPA | **方案 A（推荐）：IPA → X-SAMPA（ASCII）后训练** |
| 非拉丁原生文字（彝文音节、韩文、阿拉伯、天城体） | ⚠️ 需处理 | 要么罗马化，要么扩展词表挂原文字符 |

**一句话**：任何语言，只要能把文本编成 Qwen 词表里的干净 token，就能用藏语同款流程微调；
IPA 语言的最优解是把 IPA 转成 ASCII（X-SAMPA），复用 Wylie 的成功路径，无需改模型结构。

---

## 1. 为什么“文本表示”是成败线（回顾藏语）

藏语实验的关键决策（§2.1 实测）：

- 藏文 Unicode：**0 / 13** 覆盖（全不在词表 → 拆成无意义 byte token）。
- Wylie 罗马化：**12 / 13** 覆盖（分词干净）。

→ 选 Wylie，LLM 学会 `Wylie → 藏语语义 token`，音色由零样本复刻保住。
这与 CosyVoice3 用 Qwen 分词器（非 CosyVoice2 的独立 `tokenizer.txt` 音素词表）直接相关：
**文本侧是子词 token 序列，LLM 是语言无关的——只要 token 干净，它就能学“token→语音”**。

延伸推论：Qwen 词表对拉丁/ASCII 覆盖极好（训练语料以英文/代码为主），
所以**任何 ASCII 罗马化文本天然可切干净 token**，这正是拉丁系语言普遍可训的根本原因。

---

## 2. 拉丁可罗马化语言：统统可训

### 2.1 原理
Qwen2.5 词表 Latin/ASCII 覆盖完整，罗马化串会被切成紧凑、稳定的 subword。
LLM 微调目标 = 学会「该罗马化 → 该语言的语义 token」，说话人仍由零样本复刻决定。
覆盖示例（藏语已验证 12/13；越南语、日语 romaji、汉语拼音等 ASCII 部分必然 100%）。

### 2.2 适用语言举例
藏语(Wylie)、日语(romaji)、越南语(quốc ngữ)、汉语(拼音)、韩语(罗马字)、
印尼语、斯瓦希里语、泰语(罗马化)…… 只要是“能写成拉丁字母的音转写”都行。

### 2.3 罗马化质量建议（影响音质与所需数据量）
- **优先音位级罗马化**（pinyin、X-SAMPA）：直接编码发音，LLM 少走“拼写→读音”的歧义，
  理论上更省数据、音更准。
- 避免歧义拼写罗马化（如英语式拼写）当训练文本——除非数据足够多让 LLM 自己学 G2P。
- 带变音符的语言（越南语 `ắ`、`ế`；非洲声调语 `ɗ` 等）：先跑 §5 的覆盖率检查。
  越南语常见字符通常在词表；若个别变音符不在，做“去变音符归一”或自定义 ASCII 码即可。

### 2.4 训练改动
只需把 `prepare_data.py` 产出的 parquet `text` 字段设为罗马化串；
`instruct`、`audio_data`、训练 yaml、推理脚本与藏语完全一致。无需动模型结构。

---

## 3. IPA 语言（以彝语为例）：实测覆盖 + 映射方案

### 3.1 实测（底座 `Fun-CosyVoice3-0.5B/CosyVoice-BlankEN/vocab.json`）

```
IPA Extensions 块(U+0250-02AF) 覆盖: 0/96
彝语相关 IPA 样例覆盖:            (1, 32)   # 仅 1 个是 ASCII 字母，IPA 符号全缺
彝语 IPA 示意句覆盖:              (5, 15)   # 5 个为 ASCII 字母，余为 tone/diacritic 缺失
X-SAMPA(ASCII) 示意覆盖:          (22, 22)  # 100%
彝文音节块(U+A000-A0FF) 覆盖:    0/256
```

结论：

> 易错点：IPA 符号在 Unicode 里的**官方名字**叫 “LATIN SMALL LETTER ESH(ʃ) / ENG(ŋ) /
> OPEN E(ɛ) …”，名字带 “LATIN” 极易让人误以为它们在拉丁词表里。实测：**ʃ=False、ŋ=False、
> ɛ=False、ʒ=False、ʔ=False、ɡ=False** —— 核心 IPA 符号几乎全不在 Qwen 词表。

- **核心 IPA 符号几乎都不在词表**：IPA Extensions 块（U+0250–02AF，含 ʃ ʒ ɛ ɑ ɪ ɔ ʊ ə ʔ ɡ
  ɲ ʈ ɖ ɭ ɳ ʍ ɣ ɯ ɤ ʲ ʰ ˈ ˌ ː 等 96 个）覆盖 0/96；Greek 块（含 θ）0/135。所以 ʃ、ŋ 这类
  常用符号**确实仍需二次映射到 ASCII**（X-SAMPA：`ʃ→S`、`ŋ→N`）。
- **极少数例外**：Latin-1 Supplement 里的 ð（ETH, U+00F0）、þ（THORN, U+00FE）被覆盖
  （Latin-1 覆盖 94/128）——若你的音素集用到它们，可不必映射；但 θ（希腊 θ, U+03B8）不在词表。
- **彝文音节块 0/256** 同样不在词表。
- 实际影响：直接喂原始 IPA，绝大多数符号会落到 byte-fallback（每字符 2–3 个 `<0xXX>` token），
  序列暴涨且 LLM 未见过这些 byte 模式当“音素”，学不稳、易崩音。
- **X-SAMPA 100% ASCII 覆盖** → 与 Wylie 同机制，分词干净，是最优路径；且为一致性与稳健性，
  建议把整个音素集（含 ð/þ）统一映射到 X-SAMPA，避免训练/推理代码对个别符号分支处理。

> 注：Qwen 分词器有 byte token，所以 IPA “能编码”，但是以 **byte** 形式而非**音素**形式，
> 训练代价高、效果差。我们要的是“音素级干净 token”，所以必须 ASCII 化。

### 3.2 方案 A（推荐）：IPA → X-SAMPA（ASCII 化）

X-SAMPA 是 IPA 的 **ASCII 转写标准**，1:1 可逆。常用映射：

| IPA | X-SAMPA | IPA | X-SAMPA | IPA | X-SAMPA |
|---|---|---|---|---|---|
| ʃ | `S` | ŋ | `N` | ʒ | `Z` |
| θ | `T` | ð | `D` | ʔ | `?` |
| ɛ | `E` | ɑ | `A` | ɪ | `I` |
| ʊ | `U` | ɔ | `O` | ə | `@` |
| tʰ(送气) | `t_h` | kʰ | `k_h` | ː(长) | `:` |
| ˈ(重音) | `"` | ˌ(次重) | `%` | ɡ | `g` |

- **数据准备**：语料 IPA → 写 `ipa2xsampa.py` 转 X-SAMPA → 写入 parquet `text`。
- **训练**：复用 `cosyvoice3.yaml`（只换 `data.list`）。
- **推理**：输入同样必须是 X-SAMPA 串，且 `text_frontend=False`（不做中文归一化）。
- **优势**：音位级、ASCII 全覆盖、不改模型、与藏语 Wylie 同思路；比拼写罗马化更贴真实发音。

### 3.3 方案 B：自定义单字符 ASCII 码表（比 X-SAMPA 更紧凑）

X-SAMPA 有些是多字符（`t_h` = 2 token）。若音素集小（彝语约几十个），可自定义
“每音素一个 ASCII 字符”的码表（如 `T`→tʰ、`S`→ʃ、`5`→˥），token 更稳更省。
适合音素集封闭的语言，代价是要自己维护码表与转写脚本。

### 3.4 方案 C：扩展词表，直接挂 IPA token（不推荐首选）

给 Qwen tokenizer 新增 IPA 符号为 token，并 `resize_token_embeddings` 给新行随机初始化，
再从底座热启动微调把新 embedding 训熟。

- 优点：保留 IPA 可读性，可复用现成 IPA 转写工作流。
- 缺点：**改了模型结构**，需额外显存、需较多数据把新 token 训熟，风险高于 A/B。
- 适用：数据极充足、且团队强依赖 IPA 工作流时。

### 3.5 方案 D（不推荐）：直接喂 IPA 走 byte-fallback

能跑，但序列膨胀数倍、音素边界丢失，仅适合做“下限 baseline”，生产不可用。

---

## 4. 通用决策树

```
你的语言文本能写成 ASCII 罗马化吗？
├─ 能（且是音位级更好）
│   └─ 直接训：text=罗马化串，text_frontend=False。  ← 藏语/Wylie 同款
└─ 不能（原生非拉丁文字，或你手里只有 IPA 转写）
    ├─ 有 IPA 转写 → 转 X-SAMPA（方案 A）/ 自定义码表（方案 B）
    └─ 有原生文字（彝文/韩文/阿拉伯…）
        ├─ 能罗马化 → 罗马化后训
        └─ 想保留原文字 → 扩展词表挂原文字符（方案 C）
```

---

## 5. 实施步骤（以彝语 + X-SAMPA 为例）

```bash
# 1) 准备 IPA 转写语料（wav + ipa tsv）
# 2) IPA -> X-SAMPA
python ipa2xsampa.py in.tsv out_xsampa.tsv
# 3) prepare_data.py：text = X-SAMPA 串；instruct 同藏语（或情绪指令，见 §6）
# 4) 训练：复用 examples/tibetan/run_train_12g.sh（只换 data.list / 输出目录）
# 5) 推理：infer 脚本输入 X-SAMPA，text_frontend=False
```

### 覆盖率自检脚本（训练/推理前必跑，确认文本可干净分词）

```python
import json, os
vocab = json.load(open('pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN/vocab.json'))
vset = set(vocab.keys())
def cov(text):
    chars = [c for c in text if not c.isspace()]
    return sum(1 for c in chars if c in vset), len(chars)
# 把待训文本代表片段粘进来，覆盖应接近 100%
print(cov("你的 X-SAMPA / 罗马化 文本样例"))
```

---

## 6. 与“情绪指令(instruct)”的结合

文本表示解决“说什么语言/音素”，`instruct` 解决“用什么情绪/风格说”（藏语实验 §2.3）。
二者正交、可叠加：
- `text` = X-SAMPA / 罗马化（语言/音素层）
- `instruct` = `请用彝语里一种特别悲伤的情绪表达。<|endofprompt|>`（风格层）

要做情绪可控，需**按情绪给每条样本打标 + instruct 随样本变化**地训练
（当前藏语实验用统一中性 instruct，只学会了“说藏语”，未分化情绪）。
这是低资源语言 TTS 的下一步自然方向。

---

## 7. 风险提示

- **byte-fallback 序列膨胀**：若误用方案 D，需调大 `token_max_length` / `max_frames_in_batch`，
  否则触发 `filter` 截断、丢句。
- **码表一致性**：训练与推理必须用同一套罗马化/X-SAMPA 码表，否则发音崩坏。
- **扩展词表方案 C**：新增 embedding 行随机初始化，需足够数据训熟，且占用额外显存/磁盘。
- **磁盘**：每个 epoch 检查点 ~6.6 GB（0.5B + ZeRO-2），训练前预留空间，或降低 `save_interval`。
