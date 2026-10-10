#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""凉山彝语拼音(Liangshan Romanization) -> IPA（彝语 04-iiiwav 子集用）。

04-iiiwav 是圣经朗读切分，文本为凉山彝语拼音（如 "yi su cyt"、"yur"、"lep"），
音节末尾的 t/x/p 为声调字母，无标记为中平调。本脚本按标准凉山拼音方案把
声母 / 韵母 / 声调映射到 IPA，并把声调写成“上标数字对”(55/34/21/33)，以便复用
ipa2xsampa.py 统一转成 X-SAMPA。

重要说明:
  - 04 是凉山诺苏语，06 是楚雄彝语(梅葛)，二者是不同方言、声调系统不同
    （楚雄用 55/33/31/22/34，凉山用 55/34/21/33）。本脚本产物仅供“统一表示”，
    数字调值不可直接跨方言比较；严格对齐需母语者校验。
  - hm/hn 暂按 m/n 近似（清化鼻音），hl 按 ɬ；nz/nr/nj 等前鼻化塞音按近似值
    处理（见 INITIAL_IPA），均在 README 标注。
  - 解析失败的 token 原样保留（仍是 ASCII，Qwen 可切分）。

用法:
  python roman2ipa.py "yur lep"          # 输出 IPA（上标数字调）
  python roman2ipa.py --selfcheck        # 打印 04 语料里无法解析的 token 样例
"""
import sys
import argparse

# 声母（最长优先匹配）
INITIALS = [
    'nb', 'bb', 'hm', 'hn', 'nd', 'dd', 'hl', 'hx', 'ng', 'nz', 'zz',
    'zh', 'ch', 'rr', 'nr', 'nj', 'jj', 'ny', 'sh', 'ss',
    'b', 'p', 'm', 'f', 'v', 'd', 't', 'n', 'l', 'g', 'k', 'h',
    'w', 'z', 'c', 's', 'r', 'j', 'q', 'x', 'y',
]
INITIAL_IPA = {
    'nb': 'ᵐb', 'bb': 'b', 'hm': 'm', 'hn': 'n', 'nd': 'ⁿd', 'dd': 'd',
    'hl': 'ɬ', 'hx': 'ɕ', 'ng': 'ŋ', 'nz': 'ⁿts', 'zz': 'dz',
    'zh': 'tʂ', 'ch': 'tʂʰ', 'rr': 'dʐ', 'nr': 'ⁿtʂ', 'nj': 'ⁿtɕ', 'jj': 'dʑ',
    'ny': 'ɲ', 'sh': 'ʂ', 'ss': 'z',
    'b': 'p', 'p': 'pʰ', 'm': 'm', 'f': 'f', 'v': 'v',
    'd': 't', 't': 'tʰ', 'n': 'n', 'l': 'l', 'g': 'k', 'k': 'kʰ', 'h': 'x',
    'w': 'w', 'z': 'ts', 'c': 'tsʰ', 's': 's', 'r': 'ʐ',
    'j': 'tɕ', 'q': 'tɕʰ', 'x': 'ɕ', 'y': 'j',
}
# 韵母
FINALS = ['ie', 'uo', 'ur', 'yr', 'i', 'e', 'u', 'o', 'y', 'a']
FINAL_IPA = {
    'ie': 'ɛ', 'uo': 'ɔ', 'ur': 'u', 'yr': 'ɿ',
    'i': 'i', 'e': 'ɯ', 'u': 'u', 'o': 'o', 'y': 'ɿ', 'a': 'a',
}
# 声调：末位 t/x/p 为调字母；无标记=中平 33
TONE_LETTER = {'t': '⁵⁵', 'x': '³⁴', 'p': '²¹'}
TONE_DEFAULT = '³³'


def convert_token(tok):
    if not tok:
        return tok
    tone = TONE_DEFAULT
    stem = tok
    if tok[-1] in TONE_LETTER:
        tone = TONE_LETTER[tok[-1]]
        stem = tok[:-1]
    ini = next((i for i in INITIALS if stem.startswith(i)), None)
    if ini is None:
        return tok                      # 无法解析，保留原样
    fin = stem[len(ini):]
    if fin not in FINAL_IPA:
        return tok                      # 无法解析，保留原样
    return INITIAL_IPA[ini] + FINAL_IPA[fin] + tone


def convert_sentence(sent):
    return ' '.join(convert_token(t) for t in sent.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('text', nargs='?', default=None)
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--csv', default='/root/autodl-tmp/CosyVoice/yi-corpus/metadata.csv')
    args = ap.parse_args()

    if args.selfcheck:
        import csv
        import collections
        rows = list(csv.DictReader(open(args.csv, encoding='utf-8-sig')))
        toks = [t for r in rows if r['source'] == '04-iiiwav' for t in r['text'].split()]
        fail = collections.Counter(t for t in toks if convert_token(t) == t)
        print('04 tokens:', len(toks), 'unique:', len(set(toks)))
        print('unparsed (kept as-is):', fail.most_common(40))
        return

    if args.text:
        print(convert_sentence(args.text))
    else:
        for line in sys.stdin:
            print(convert_sentence(line.rstrip('\n')))


if __name__ == '__main__':
    main()
