#!/usr/bin/env bash
# =============================================================================
# CosyVoice3 新服务器部署 / 预训练模型下载索引
#
#   模型仓库 : FunAudioLLM/Fun-CosyVoice3-0.5B-2512  (ModelScope，19 个文件，9.75 GB)
#   落地目录 : $COSYVOICE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B
#
# 用法:
#   bash scripts/setup_new_server.sh list     # 只打印下载索引，不做任何下载
#   bash scripts/setup_new_server.sh full     # 下载全部 19 个文件 (9.75 GB)  ← 默认
#   bash scripts/setup_new_server.sh slim     # 只下训练/推理必需 (6.40 GB，省 3.35 GB)
#   bash scripts/setup_new_server.sh verify   # 只校验已下载文件的完整性（按字节数）
#   bash scripts/setup_new_server.sh code     # 只准备代码（子模块等，不下载模型）
#
# 环境变量:
#   COSYVOICE_ROOT  仓库根目录   默认 /root/autodl-tmp/CosyVoice
#   MAX_WORKERS     并发下载线程 默认 8（ModelScope 支持多线程分块下载）
#   PYTHON          python 解释器 默认 python
#   MODELSCOPE_ENDPOINT  可选，自定义镜像站
# =============================================================================
set -euo pipefail

ROOT="${COSYVOICE_ROOT:-/root/autodl-tmp/CosyVoice}"
MODEL_REPO="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODEL_DIR="$ROOT/pretrained_models/Fun-CosyVoice3-0.5B"
MODE="${1:-full}"
WORKERS="${MAX_WORKERS:-8}"
PY="${PYTHON:-python}"

# -----------------------------------------------------------------------------
# 下载索引：路径 | 字节数 | full | slim | 说明
#   full=1 表示 full 模式下载；slim=1 表示 slim 模式下载
# -----------------------------------------------------------------------------
read -r -d '' INDEX <<'EOF' || true
llm.pt|2024669519|1|1|LLM 初始权重 —— 微调的起点
llm.rl.pt|2024682701|1|0|RL 强化版 LLM，不做 RL 训练就不需要
flow.pt|1329116148|1|1|Flow Matching 模块，推理/评估需要
flow.decoder.estimator.fp32.onnx|1326216933|1|0|仅 load_trt=True（TensorRT 加速）时用
hift.pt|83202622|1|1|HiFi-GAN 声码器，推理需要
campplus.onnx|28303423|1|1|说话人 embedding 提取，训练在线提取也要
speech_tokenizer_v3.batch.onnx|969451579|1|1|语音 token 提取(batch) —— 训练链路用这个
speech_tokenizer_v3.onnx|969451503|1|1|语音 token 提取(单条) —— 推理链路用这个
cosyvoice3.yaml|6934|1|1|模型结构配置
config.json|2|1|1|占位配置
configuration.json|47|1|1|占位配置
README.md|11982|1|1|模型说明
CosyVoice-BlankEN/model.safetensors|988097824|1|1|Qwen2 文本编码器 —— qwen_pretrain_path
CosyVoice-BlankEN/vocab.json|2776833|1|1|Qwen 词表
CosyVoice-BlankEN/merges.txt|1402109|1|1|BPE merges
CosyVoice-BlankEN/config.json|659|1|1|Qwen 模型配置
CosyVoice-BlankEN/generation_config.json|242|1|1|生成配置
CosyVoice-BlankEN/tokenizer_config.json|1287|1|1|tokenizer 配置
asset/dingding.png|*|1|0|仓库配图，纯展示，可不下
EOF

# slim 模式允许的文件（glob 模式，交给 modelscope 过滤）
SLIM_PATTERNS='["llm.pt","flow.pt","hift.pt","campplus.onnx","speech_tokenizer_v3.batch.onnx","speech_tokenizer_v3.onnx","cosyvoice3.yaml","config.json","configuration.json","README.md","CosyVoice-BlankEN/*"]'

hr() { printf '%s\n' "---------------------------------------------------------------------------"; }
info() { printf '\033[32m[%s]\033[0m %s\n' "$1" "$2"; }
warn() { printf '\033[33m[%s]\033[0m %s\n' "$1" "$2"; }
die() { printf '\033[31m[ERROR]\033[0m %s\n' "$1" >&2; exit 1; }

# -----------------------------------------------------------------------------
cmd_list() {
    printf '\n\033[1mCosyVoice3 预训练模型下载索引\033[0m\n'
    hr
    printf '%-42s %12s  %-6s %-6s %s\n' "文件" "大小" "full" "slim" "说明"
    hr
    local total_full=0 total_slim=0
    while IFS='|' read -r path size f s desc; do
        [ -z "${path:-}" ] && continue
        local human
        if [ "$size" = "*" ]; then human="?"; else
            human="$(awk -v b="$size" 'BEGIN{printf (b>=1e9?"%.2f GB":(b>=1e6?"%.1f MB":"%.0f B")), (b>=1e9?b/1e9:(b>=1e6?b/1e6:b))}')"
            [ "$f" = "1" ] && total_full=$((total_full + size))
            [ "$s" = "1" ] && total_slim=$((total_slim + size))
        fi
        printf '%-42s %12s  %-6s %-6s %s\n' "$path" "$human" \
            "$([ "$f" = 1 ] && echo '✓' || echo '-')" \
            "$([ "$s" = 1 ] && echo '✓' || echo '-')" "$desc"
    done <<< "$INDEX"
    hr
    awk -v a="$total_full" -v b="$total_slim" \
        'BEGIN{printf "合计: full = %.2f GB   slim = %.2f GB   可省 %.2f GB\n", a/1e9, b/1e9, (a-b)/1e9}'
    hr
    printf '仓库: %s\n落地: %s\n\n' "$MODEL_REPO" "$MODEL_DIR"
}

# -----------------------------------------------------------------------------
cmd_code() {
    info "code" "检查代码与子模块"
    [ -d "$ROOT/.git" ] || die "$ROOT 不是 git 仓库。请先: git clone -b dev git@github.com:feelins/CosyVoice.git $ROOT"

    local sub="$ROOT/third_party/Matcha-TTS"
    if [ ! -f "$sub/setup.py" ] && [ ! -f "$sub/pyproject.toml" ]; then
        warn "code" "子模块 third_party/Matcha-TTS 为空，正在初始化..."
        # 子模块原地址是 github，国内可用镜像替换加速
        git -C "$ROOT" -c url."https://gitclone.com/github.com/".insteadOf="https://github.com/" \
            submodule update --init --depth 1 third_party/Matcha-TTS
    fi
    info "code" "子模块 OK: $(git -C "$sub" rev-parse --short HEAD 2>/dev/null || echo '?')"

    if [ ! -f "$ROOT/examples/teochew/data/train.data.list" ]; then
        warn "code" "训练数据清单缺失: examples/teochew/data/  (被 .gitignore 忽略，需单独生成或传输)"
    fi
}

# -----------------------------------------------------------------------------
cmd_download() {
    local mode="$1"
    [ -d "$ROOT" ] || die "$ROOT 不存在，请先 clone 仓库"

    local free_kb
    free_kb="$(df -Pk "$ROOT" | awk 'NR==2{print $4}')"
    local free_gb=$((free_kb / 1024 / 1024))
    local need_gb=10
    [ "$mode" = "slim" ] && need_gb=7
    info "disk" "目标盘可用 ${free_gb} GB，需要约 ${need_gb} GB"
    [ "$free_gb" -ge "$need_gb" ] || die "空间不足，请先清理或把 COSYVOICE_ROOT 指到数据盘"

    mkdir -p "$(dirname "$MODEL_DIR")"

    info "download" "模式=$mode  并发=$WORKERS  仓库=$MODEL_REPO"
    info "download" "目标=$MODEL_DIR"
    warn "download" "首次下载约 ${need_gb} GB，ModelScope 支持断点续传，中断后重跑本脚本即可"

    MODE="$mode" WORKERS="$WORKERS" MODEL_REPO="$MODEL_REPO" \
    MODEL_DIR="$MODEL_DIR" SLIM_PATTERNS="$SLIM_PATTERNS" PY="$PY" "$PY" - <<'PYEOF'
import os, sys, time

mode = os.environ["MODE"]
workers = int(os.environ["WORKERS"])
repo = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]
patterns = None if mode == "full" else __import__("json").loads(os.environ["SLIM_PATTERNS"])

try:
    from modelscope import snapshot_download
except ImportError:
    sys.exit("未安装 modelscope，请先: pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")

kwargs = dict(local_dir=local_dir, max_workers=workers)
if patterns:
    kwargs["allow_patterns"] = patterns
ep = os.environ.get("MODELSCOPE_ENDPOINT")
if ep:
    kwargs["endpoint"] = ep

t0 = time.time()
path = snapshot_download(repo, **kwargs)
print(f"\n[download] 完成: {path}")
print(f"[download] 耗时 {time.time() - t0:.1f}s")
PYEOF
}

# -----------------------------------------------------------------------------
cmd_verify() {
    [ -d "$MODEL_DIR" ] || die "$MODEL_DIR 不存在"
    info "verify" "按字节数校验 $MODEL_DIR"

    MODEL_DIR="$MODEL_DIR" INDEX="$INDEX" MODE="$MODE" "$PY" - <<'PYEOF'
import os, sys

model_dir = os.environ["MODEL_DIR"]
mode = os.environ["MODE"]
rows = [l for l in os.environ["INDEX"].splitlines() if l.strip()]

need = lambda f, s: (f == "1") if mode == "full" else (s == "1")
ok = bad = miss = skip = 0
for row in rows:
    path, size, full, slim, desc = row.split("|", 4)
    p = os.path.join(model_dir, path)
    if not os.path.exists(p):
        if need(full, slim):
            print(f"  [缺失] {path}"); miss += 1
        else:
            skip += 1
        continue
    actual = os.path.getsize(p)
    if size == "*":
        print(f"  [ OK ] {path}  ({actual/1e6:.2f} MB)"); ok += 1
    elif actual == int(size):
        print(f"  [ OK ] {path}  ({actual/1e9:.2f} GB)"); ok += 1
    else:
        print(f"  [不符] {path}  期望 {int(size)} 实际 {actual}"); bad += 1

print(f"\n  校验结果: OK={ok}  不符={bad}  缺失={miss}  跳过(非必需)={skip}")
if bad or miss:
    print("\n  修复: 重跑  bash scripts/setup_new_server.sh full  (ModelScope 自动续传补齐)")
    sys.exit(1)
print("  全部文件完整 ✓")
PYEOF
}

# -----------------------------------------------------------------------------
case "$MODE" in
    list)        cmd_list ;;
    code)        cmd_code ;;
    verify)      cmd_verify ;;
    full|slim)   cmd_code; cmd_download "$MODE"; cmd_verify ;;
    -h|--help)   sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' ;;
    *)           die "未知模式: $MODE（可用: list | full | slim | verify | code）" ;;
esac
