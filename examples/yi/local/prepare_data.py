#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""彝语(IPA) -> CosyVoice3 训练 parquet（初稿，复用藏语管线结构）。

流程:
  IPA tsv --(ipa2xsampa)--> X-SAMPA 文本 --> parquet(text = X-SAMPA)

输入(约定):
  /root/autodl-tmp/yi/{train,valid,test}-ipa.tsv
    表头: idx\tpath\tipa
    数据: idx\twav名\tiPA音标句
  音频: /root/autodl-tmp/yi_wavs/wav/<path>

输出:
  data/train/parquet/{parquet_000000000.tar, data.list}
  data/cv/    (valid)
  data/test/  (test)
  data/train.data.list -> train/parquet/data.list   (run_train_12g.sh 用)
  data/dev.data.list   -> cv/parquet/data.list

说明:
  - text 用 X-SAMPA（IPA 的 ASCII 版；底座 Qwen 词表对 IPA 覆盖 ~0，对 X-SAMPA 100%）。
  - 训练/推理均须 text_frontend=False，且推理输入同样是 X-SAMPA。
  - instruct 固定「请用彝语表达」；情绪可控版见分析文档 §6（instruct 随样本变化）。
  - spk 从文件名提取（按 'OTR002-01' 这类分段；无则归 'yi'）。

DRAFT: 尚未验证真实彝语语料字段，先按藏语结构搭好；拿到 IPA tsv + wav 后微调 WAV_DIR / 字段解析即可。
"""
import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipa2xsampa import convert as ipa2xsampa

WAV_DIR = '/root/autodl-tmp/yi_wavs/wav'
INSTRUCT = 'You are a helpful assistant. 请用彝语表达。<|endofprompt|>'


def spk_of(wav_path):
    base = os.path.basename(wav_path).replace('.wav', '')
    toks = base.split('-')
    if len(toks) >= 3:
        return toks[1] + '-' + toks[2]
    return 'yi'


def parse_tsv(tsv):
    utt2wav, utt2text = {}, {}
    missing = 0
    with open(tsv, encoding='utf-8') as f:
        f.readline()  # 跳过表头
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            wav, ipa = parts[1].strip(), parts[2].strip()
            if wav == 'path' or ipa == 'sentence':
                continue
            if not ipa:
                continue
            new_wav = os.path.join(WAV_DIR, wav)
            utt = os.path.basename(wav).replace('.wav', '')
            if not os.path.exists(new_wav):
                missing += 1
                continue
            utt2wav[utt] = new_wav
            utt2text[utt] = ipa2xsampa(ipa)   # IPA -> X-SAMPA
    return utt2wav, utt2text, missing


def build_parquet(d, utts, utt2wav, utt2text):
    parquet_dir = os.path.join(d, 'parquet')
    os.makedirs(parquet_dir, exist_ok=True)
    columns = {
        'utt': utts,
        'audio_data': [open(utt2wav[u], 'rb').read() for u in utts],
        'wav': [utt2wav[u] for u in utts],
        'text': [utt2text[u] for u in utts],
        'spk': [spk_of(utt2wav[u]) for u in utts],
        'instruct': [INSTRUCT] * len(utts),
    }
    pf = os.path.join(parquet_dir, 'parquet_000000000.tar')
    pq.write_table(pa.table(columns), pf)
    with open(os.path.join(parquet_dir, 'data.list'), 'w', encoding='utf-8') as f:
        f.write(pf + '\n')
    return pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/autodl-tmp')
    ap.add_argument('--des_dir', default='data')
    args = ap.parse_args()

    sets = [('train', 'train-ipa.tsv'),
            ('cv', 'valid-ipa.tsv'),
            ('test', 'test-ipa.tsv')]
    for name, fn in sets:
        tsv = os.path.join(args.root, fn)
        if not os.path.exists(tsv):
            print('[skip]', tsv)
            continue
        utt2wav, utt2text, missing = parse_tsv(tsv)
        utts = sorted(utt2wav.keys())
        d = os.path.join(args.des_dir, name)
        os.makedirs(d, exist_ok=True)
        pf = build_parquet(d, utts, utt2wav, utt2text)
        spks = sorted(set(spk_of(utt2wav[u]) for u in utts))
        print('[yi prep] {}: {} 句, 缺失音频 {} 句, 说话人 {} -> {}'.format(
            name, len(utts), missing, spks, pf))

    os.makedirs(args.des_dir, exist_ok=True)
    for top, sub in [('train.data.list', 'train/parquet/data.list'),
                     ('dev.data.list', 'cv/parquet/data.list')]:
        tgt = os.path.join(args.des_dir, top)
        if os.path.islink(tgt) or os.path.exists(tgt):
            os.remove(tgt)
        os.symlink(sub, tgt)
    print('[yi prep] 软链 data/train.data.list, data/dev.data.list 已创建')


if __name__ == '__main__':
    main()
