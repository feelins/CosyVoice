#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IPA -> X-SAMPA 转写（彝语 / 通用低资源语言 · 初稿）。

把 IPA 音标转成纯 ASCII 的 X-SAMPA，使 CosyVoice3 的 Qwen 分词器能干净切分
（实测：IPA Extensions 块 U+0250-02AF 覆盖 0/96，X-SAMPA 100% ASCII 覆盖）。

用法:
  # 文本文件: 每行一句 IPA
  python ipa2xsampa.py in.ipa.txt out.xsampa.txt
  # 管道
  cat a.ipa | python ipa2xsampa.py > a.xsampa
  # 检查某语料里所有 IPA 符号的覆盖情况（打印缺失符号，便于补全 MAP）
  python ipa2xsampa.py --check corpus.ipa.txt

训练/推理约定:
  - parquet 的 text 字段 = X-SAMPA 串
  - 推理时同样输入 X-SAMPA，且 CosyVoice 必须 text_frontend=False

DRAFT: MAP 覆盖彝语/通用 IPA 常见音素，请先用 --check 跑你的语料，
未覆盖的符号会打印告警，按需把新音素补进 MAP（键=IPA 单字符，值=X-SAMPA 串，
可多字符如送气 't_h'）。
"""
import sys
import argparse

# ---- IPA(单字符) -> X-SAMPA(可多字符) 初稿映射 ----
_VOWELS = {
    'a': 'a', 'i': 'i', 'u': 'u', 'e': 'e', 'o': 'o',
    'ɑ': 'A', 'ɒ': 'Q', 'ɔ': 'O', 'ɛ': 'E', 'ə': '@', 'ɜ': '3',
    'ɪ': 'I', 'ʊ': 'U', 'ʌ': 'V', 'ɨ': '1', 'ɯ': 'M', 'ɤ': 'O',
    'ø': '2', 'y': 'y', 'æ': '{', 'ʉ': '}', 'ɘ': '"', 'ɵ': '8',
}
_CONSONANTS = {
    'ʃ': 'S', 'ʒ': 'Z', 'θ': 'T', 'ð': 'D', 'ŋ': 'N', 'ɲ': 'J',
    'ʂ': 's`', 'ʐ': 'z`', 'ʈ': 't`', 'ɖ': 'd`', 'ɭ': 'l`', 'ɳ': 'n`',
    'ɡ': 'g', 'ɣ': 'G', 'ʔ': '?', 'ɦ': 'h', 'ʕ': 'H', 'χ': 'x',
    'ʁ': 'R', 'ɻ': 'r`', 'ʎ': 'j', 'ʝ': 'j`', 'ç': 'c', 'ɕ': 's`',
    'ʑ': 'z`', 'ɸ': 'p', 'β': 'B', 'ʍ': 'W', 'ʋ': 'P', 'ɱ': 'F',
    'ɢ': 'G`', 'ɠ': 'g`', 'ʡ': '>', 'ʢ': '<', 'ʜ': 'C', 'ʟ': 'L',
}
_MODIFIERS = {
    'ʰ': '_h', 'ʲ': '_j', 'ʷ': '_w', 'ˈ': '"', 'ˌ': '%', 'ː': ':', 'ˑ': ':',
}
# 声调: 彝语/通用调值字母 -> 数字 1..5（便于切成独立 token；要 X-SAMPA 风格改这里）
_TONES = {'˥': '5', '˦': '4', '˧': '3', '˨': '2', '˩': '1'}
# Latin-1 里底座词表已覆盖的符号，保留原样（也可改为统一映射，这里留原字符）
_LATIN1_KEEP = {'ð': 'ð', 'þ': 'þ'}

MAP = {**_VOWELS, **_CONSONANTS, **_MODIFIERS, **_TONES, **_LATIN1_KEEP}

_PASS = set(' .,!?;:—…')


def convert(s):
    """IPA 字符串 -> X-SAMPA 字符串。未覆盖符号保留原字符并打印告警。"""
    out = []
    for ch in s:
        if ch in MAP:
            out.append(MAP[ch])
        elif ch in _PASS or ch.isspace():
            out.append(ch)
        else:
            out.append(ch)
            sys.stderr.write('[WARN] 未覆盖 IPA 符号: %r (U+%04X)\n' % (ch, ord(ch)))
    return ''.join(out)


def check_coverage(path):
    miss = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            for ch in line:
                if ch not in MAP and ch not in _PASS and not ch.isspace():
                    miss.add(ch)
    print('缺失(未覆盖)符号:', sorted(miss, key=lambda c: ord(c)))
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', nargs='?', default=None, help='输入 IPA 文件')
    ap.add_argument('output', nargs='?', default=None, help='输出 X-SAMPA 文件')
    ap.add_argument('--check', metavar='FILE', help='仅检查该文件 IPA 覆盖情况')
    args = ap.parse_args()

    if args.check:
        check_coverage(args.check)
        return

    inp = open(args.input, encoding='utf-8') if args.input else sys.stdin
    out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
    for line in inp:
        out.write(convert(line.rstrip('\n')) + '\n')


if __name__ == '__main__':
    main()
