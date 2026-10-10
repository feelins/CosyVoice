#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 06-only 的 test parquet 生成推理样例 tsv。

列: idx \t wav(绝对路径) \t xsampa文本
首行作为参考音色(prompt), 其余作为目标句; origin 组用各目标句原始录音对照。

用法:
  cd /root/autodl-tmp/CyVoice
  python yi_eval/gen_samples_tsv.py --n 6 --out yi_eval/samples.tsv
"""
import os
import sys
import argparse
import pyarrow.parquet as pq
import glob

ROOT = '/root/autodl-tmp/CosyVoice'
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', default='yi-corpus/data/test/parquet')
    ap.add_argument('--out', default='yi_eval/samples.tsv')
    ap.add_argument('--n', type=int, default=6)
    args = ap.parse_args()

    fs = glob.glob(os.path.join(args.parquet, 'parquet_*.tar'))
    assert fs, '未找到 parquet: {}'.format(args.parquet)
    t = pq.read_table(fs[0])
    wavs = t.column('wav').to_pylist()
    texts = t.column('text').to_pylist()
    n = min(args.n, len(wavs))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for i in range(n):
            f.write('{}\t{}\t{}\n'.format(i, wavs[i], texts[i]))
    print('写出', args.out, '共', n, '行(首行 idx=0 为 prompt, 其余为目标句)')


if __name__ == '__main__':
    main()
