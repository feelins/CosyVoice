#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""彝语 X-SAMPA tsv -> CosyVoice3 训练 parquet（复用藏语管线结构）。

输入: yi-corpus/processed/splits/{train,valid,test}.tsv
      表头: idx\twav\txsampa   数据: <id>\t<wav相对路径>\t<xsampa>
音频: yi-corpus/wavs/<wav>

输出: yi-corpus/data/{train,cv,test}/parquet/{parquet_000000000.tar, data.list}
      yi-corpus/data/train.data.list -> train/parquet/data.list
      yi-corpus/data/dev.data.list   -> cv/parquet/data.list

约定: text = X-SAMPA；instruct 固定「请用彝语表达」；推理 text_frontend=False。
spk 在 CosyVoice3 中由音频在线提取（use_spk_embedding=False），此处统一填 'yi'。
"""
import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

WAV_ROOT = '/root/autodl-tmp/CosyVoice/yi-corpus'  # metadata 里 wav 字段为 "wavs/xxx.wav"
ROOT = '/root/autodl-tmp/CosyVoice/yi-corpus'
INSTRUCT = 'You are a helpful assistant. 请用彝语表达。<|endofprompt|>'


def parse_tsv(tsv):
    utt2wav, utt2text, missing = {}, {}, 0
    with open(tsv, encoding='utf-8') as f:
        f.readline()                      # 跳过表头
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            utt, wav, xs = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if wav == 'wav' or xs == 'xsampa':
                continue
            if not xs:
                continue
            full = os.path.join(WAV_ROOT, wav)
            if not os.path.exists(full):
                missing += 1
                continue
            utt2wav[utt] = full
            utt2text[utt] = xs
    return utt2wav, utt2text, missing


def build_parquet(d, utts, utt2wav, utt2text):
    parquet_dir = os.path.join(d, 'parquet')
    os.makedirs(parquet_dir, exist_ok=True)
    columns = {
        'utt': utts,
        'audio_data': [open(utt2wav[u], 'rb').read() for u in utts],
        'wav': [utt2wav[u] for u in utts],
        'text': [utt2text[u] for u in utts],
        'spk': ['yi' for u in utts],
        'instruct': [INSTRUCT] * len(utts),
    }
    pf = os.path.join(parquet_dir, 'parquet_000000000.tar')
    pq.write_table(pa.table(columns), pf)
    with open(os.path.join(parquet_dir, 'data.list'), 'w', encoding='utf-8') as f:
        f.write(pf + '\n')
    return pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default=os.path.join(ROOT, 'processed', 'splits'))
    ap.add_argument('--data', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--which', default='all', choices=['all', '06'])
    args = ap.parse_args()
    tag = '' if args.which == 'all' else '_06'
    sets = [('train', 'train' + tag), ('cv', 'valid' + tag), ('test', 'test' + tag)]
    for name, fn in sets:
        tsv = os.path.join(args.splits, fn + '.tsv')
        if not os.path.exists(tsv):
            print('[skip]', tsv)
            continue
        utt2wav, utt2text, missing = parse_tsv(tsv)
        utts = sorted(utt2wav.keys())
        d = os.path.join(args.data, name)
        os.makedirs(d, exist_ok=True)
        pf = build_parquet(d, utts, utt2wav, utt2text)
        print('[yi prep] {}: {} 句, 缺失音频 {} -> {}'.format(name, len(utts), missing, pf))
    os.makedirs(args.data, exist_ok=True)
    for top, sub in [('train.data.list', 'train/parquet/data.list'),
                     ('dev.data.list', 'cv/parquet/data.list')]:
        tgt = os.path.join(args.data, top)
        if os.path.islink(tgt) or os.path.exists(tgt):
            os.remove(tgt)
        os.symlink(sub, tgt)
    print('[yi prep] 软链 data/train.data.list, data/dev.data.list 已创建')


if __name__ == '__main__':
    main()
