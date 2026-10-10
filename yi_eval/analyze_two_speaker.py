#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探查 06(楚雄梅葛) 语料里“一句话前女后男”的普遍程度与切换位置。

方法: 用 pyworld 提取每句 F0 基频, 比较前半段与后半段的中位 F0。
女声基频(200-400Hz) 远高于男声(100-200Hz), 故 front/back 比值 >~1.3 即视为“前高后低=女->男”切换。
不依赖 GPU, 纯 CPU, 可与训练并行。

用法:
  cd /root/autodl-tmp/CosyVoice
  python yi_eval/analyze_two_speaker.py --n 40
"""
import os
import sys
import glob
import random
import argparse
import numpy as np
import pyworld as pw
import soundfile as sf

ROOT = '/root/autodl-tmp/CosyVoice/yi-corpus'


def med(seg):
    v = seg[seg > 0]
    return float(np.median(v)) if len(v) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--wav_glob', default='wavs/06_*.wav')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    wavs = sorted(glob.glob(os.path.join(ROOT, args.wav_glob)))
    sample = random.sample(wavs, min(args.n, len(wavs)))

    rows = []
    for w in sample:
        x, fs = sf.read(w)
        if x.ndim > 1:
            x = x[:, 0]
        x = x.astype(np.float64)
        if fs != 16000:
            # 简单重采样到 16k(pyworld 友好); 这里仅做粗判, 影响不大
            import resampy
            x = resampy.resample(x, fs, 16000)
            fs = 16000
        f0, _ = pw.harvest(x, fs, f0_floor=50, f0_ceil=500)
        f0 = np.nan_to_num(f0)
        n = len(f0)
        if n < 30:
            rows.append((os.path.basename(w), 0.0, 0.0, 0.0, 'short'))
            continue
        a = f0[: n // 3]
        b = f0[2 * n // 3:]
        ma, mb = med(a), med(b)
        ratio = round(ma / mb, 2) if mb > 0 else 0.0
        tag = '女->男' if (ma > 0 and mb > 0 and ratio >= 1.25) else (
            '男->女' if (ma > 0 and mb > 0 and ratio <= 0.8) else '单一/不明')
        rows.append((os.path.basename(w), round(ma, 1), round(mb, 1), ratio, tag))

    shift = sum(1 for r in rows if r[4] == '女->男')
    mshift = sum(1 for r in rows if r[4] == '男->女')
    print('样本 {} 句 | 前高后低(女->男): {} | 前低后高(男->女): {} | 单一/不明: {}'.format(
        len(rows), shift, mshift, len(rows) - shift - mshift))
    print('{:40s} {:>8s} {:>8s} {:>7s}  {}'.format('wav', 'frontF0', 'backF0', 'ratio', '判定'))
    for r in rows:
        print('{:40s} {:>8} {:>8} {:>7}  {}'.format(r[0], r[1], r[2], r[3], r[4]))


if __name__ == '__main__':
    main()
