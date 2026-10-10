#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IPA -> X-SAMPA 转写（彝语 Chuxiong/Liangshan 微调版）。

把彝语语料里的 IPA 音标转成纯 ASCII 的 X-SAMPA，使 CosyVoice3 的 Qwen 分词器
能干净切分。彝语 IPA 大量使用 IPA Extensions 块、上标数字调值等非 ASCII 字符，
Qwen 词表覆盖很差；X-SAMPA 100% ASCII 覆盖（与藏语用 Wylie 罗马化的思路一致）。

用法:
  python ipa2xsampa.py in.ipa.txt out.xsampa.txt
  cat a.ipa | python ipa2xsampa.py > a.xsampa
  python ipa2xsampa.py --check corpus.ipa.txt   # 打印未覆盖符号

训练 / 推理约定（同潮汕、藏语零样本方言微调）:
  - parquet 的 text 字段 = X-SAMPA 串
  - 推理时同样输入 X-SAMPA，且 CosyVoice 必须 text_frontend=False
  - instruct 固定「请用彝语表达」

彝语特有处理:
  - 上标调值数字 ⁰¹²³⁴⁵⁶⁷⁸⁹ -> 0123456789（Chuxiong 用 55/33/31/22/34，
    Liangshan 用 55/34/21/33；统一成 ASCII 数字串，如 li³³ -> li33）。
  - 彝语专用 IPA：ʦ ʨ ʥ ʣ ʧ ɿ ȵ ȶ ȡ ȴ ʔ ʋ ɣ ʑ ʂ ʐ ŋ æ ə ɯ ɔ ɛ ɬ 等。
  - 送气 / 挤喉撇号 ' -> h（如 p'ə -> ph@）。
  - 大写 I -> i（语料里偶见 I⁵⁵ 应为 i⁵⁵）。
  - 全角逗号 ，-> ,。
  - 其余 ASCII 字母 / 数字 / 常见标点原样保留（Qwen 词表全覆盖）。
"""
import sys
import argparse

# ---- 彝语 IPA(单字符) -> X-SAMPA(可多字符) ----
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
    'ɬ': 'K', 'ɿ': 'z', 'ʦ': 'ts', 'ʨ': 'tc', 'ʥ': 'dj', 'ʣ': 'dz',
    'ʧ': 'tS', 'ȵ': 'J', 'ȶ': 't', 'ȡ': 'd', 'ȴ': 'l',
}
_MODIFIERS = {
    'ʰ': '_h', 'ʲ': '_j', 'ʷ': '_w', 'ˈ': '"', 'ˌ': '%', 'ː': ':', 'ˑ': ':',
}
# 上标调值数字 -> ASCII 数字（彝语调值写成上标数字对）
_SUPER_DIGITS = {
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
}
# 彝语专用小修
_SPECIAL = {
    "'": 'h',      # 送气 / 挤喉撇号 -> h（p'ə -> ph@）
    'I': 'i',      # 大写 I 误写 -> i
    '，': ',',     # 全角逗号 -> 半角
    'ᵐ': 'm',      # 前鼻化修饰 m（凉山 nb/nd 等）-> m（近似，见 roman2ipa）
    'ⁿ': 'n',      # 前鼻化修饰 n（凉山 nz/nr/nj 等）-> n（近似，见 roman2ipa）
}
MAP = {**_VOWELS, **_CONSONANTS, **_MODIFIERS, **_SUPER_DIGITS, **_SPECIAL}
_PASS = set(' .,!?;:—…、！？；：')


def convert(s):
    """IPA 字符串 -> X-SAMPA 字符串。未覆盖符号保留原字符并打印告警。"""
    out = []
    for ch in s:
        if ch in MAP:
            out.append(MAP[ch])
        elif ch in _PASS or (ch.isascii() and (ch.isalnum() or ch.isspace())):
            out.append(ch)
        else:
            out.append(ch)
            sys.stderr.write('[WARN] 未覆盖符号: %r (U+%04X)\n' % (ch, ord(ch)))
    return ''.join(out)


def check_coverage(path):
    miss = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            for ch in line:
                if ch not in MAP and ch not in _PASS and \
                        not (ch.isascii() and (ch.isalnum() or ch.isspace())):
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
