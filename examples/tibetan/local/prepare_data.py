#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把藏语 Wylie tsv 转成 CosyVoice3 训练 parquet（复用潮汕话管线）。

输入:
  /root/autodl-tmp/train-wylie.tsv    (表头: \tpath\tsentence; 数据: idx\twav\twylie)
  /root/autodl-tmp/valid-wylie.tsv
  /root/autodl-tmp/test-wylie.tsv
音频: /root/autodl-tmp/tibetan_wavs/wav/<path>

输出:
  data/train/parquet/{parquet_000000000.tar, data.list}
  data/cv/    (valid)
  data/test/  (test)
  data/train.data.list -> train/parquet/data.list   (run_train_12g.sh 用)
  data/dev.data.list   -> cv/parquet/data.list

说明:
  - text 用 Wylie 罗马化（底座 Qwen 分词器对藏文 Unicode 覆盖 0/13，对 Wylie 覆盖 12/13）。
  - instruct 固定为「请用藏语表达」，复刻潮汕话的零样本方言微调方式。
  - spk 从文件名提取（OTR002-01 女声 / OTR002-02~09 男声）。
"""
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq

WAV_DIR = '/root/autodl-tmp/tibetan_wavs/wav'
INSTRUCT = 'You are a helpful assistant. 请用藏语表达。<|endofprompt|>'


def spk_of(wav_path):
    base = os.path.basename(wav_path).replace('.wav', '')
    toks = base.split('-')
    if len(toks) >= 3:
        return toks[1] + '-' + toks[2]
    return 'tibetan'


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
            wav, text = parts[1].strip(), parts[2].strip()
            if wav == 'path' or text == 'sentence':
                continue
            if not text:
                continue
            new_wav = os.path.join(WAV_DIR, wav)
            utt = os.path.basename(wav).replace('.wav', '')
            if not os.path.exists(new_wav):
                missing += 1
                continue
            utt2wav[utt] = new_wav
            utt2text[utt] = text
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

    sets = [('train', 'train-wylie.tsv'),
            ('cv', 'valid-wylie.tsv'),
            ('test', 'test-wylie.tsv')]
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
        print('[tibetan prep] {}: {} 句, 缺失音频 {} 句, 说话人 {} -> {}'.format(
            name, len(utts), missing, spks, pf))

    os.makedirs(args.des_dir, exist_ok=True)
    for top, sub in [('train.data.list', 'train/parquet/data.list'),
                     ('dev.data.list', 'cv/parquet/data.list')]:
        tgt = os.path.join(args.des_dir, top)
        if os.path.islink(tgt) or os.path.exists(tgt):
            os.remove(tgt)
        os.symlink(sub, tgt)
    print('[tibetan prep] 软链 data/train.data.list, data/dev.data.list 已创建')


if __name__ == '__main__':
    main()
