#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 teochew-data 的 train.list 转成 CosyVoice3 训练所需的格式。

输入: /root/autodl-tmp/teochew-data/04_ChaoShan/train.list
      格式: wav路径|spk|lang|文本  (竖线分隔, 4 列)
      注意: 该文件里的 wav 路径指向 GPT-SoVITS 旧目录, 需重映射到实际音频目录。
输出:
  {des_dir}/train/{wav.scp,text,utt2spk,spk2utt,instruct,parquet/data.list}
  {des_dir}/cv/   {...同上...}
"""
import argparse
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq


def parse_train_list(src_list, wav_dir):
    utt2wav, utt2text, utt2spk = {}, {}, {}
    missing = 0
    with open(src_list, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue
            wav_orig, spk, _lang, text = parts[0], parts[1], parts[2], parts[3]
            text = text.strip()
            if not text:
                continue
            utt = os.path.basename(wav_orig).replace('.wav', '')
            new_wav = os.path.join(wav_dir, utt + '.wav')
            if not os.path.exists(new_wav):
                missing += 1
                continue
            utt2wav[utt] = new_wav
            utt2text[utt] = text
            utt2spk[utt] = spk
    return utt2wav, utt2text, utt2spk, missing


def write_kaldi_files(d, utts, utt2wav, utt2text, utt2spk, instruct):
    spk2utt = {}
    for u in utts:
        spk2utt.setdefault(utt2spk[u], []).append(u)
    with open(os.path.join(d, 'wav.scp'), 'w', encoding='utf-8') as f:
        for u in utts:
            f.write('{} {}\n'.format(u, utt2wav[u]))
    with open(os.path.join(d, 'text'), 'w', encoding='utf-8') as f:
        for u in utts:
            f.write('{} {}\n'.format(u, utt2text[u]))
    with open(os.path.join(d, 'utt2spk'), 'w', encoding='utf-8') as f:
        for u in utts:
            f.write('{} {}\n'.format(u, utt2spk[u]))
    with open(os.path.join(d, 'spk2utt'), 'w', encoding='utf-8') as f:
        for spk, us in spk2utt.items():
            f.write('{} {}\n'.format(spk, ' '.join(us)))
    if instruct:
        with open(os.path.join(d, 'instruct'), 'w', encoding='utf-8') as f:
            for u in utts:
                f.write('{} {}\n'.format(u, instruct))


def build_parquet(d, utts, utt2wav, utt2text, utt2spk, instruct):
    """自建 parquet + data.list (避开官方 make_parquet_list.py 里 spk_list 未定义的 bug)。"""
    parquet_dir = os.path.join(d, 'parquet')
    os.makedirs(parquet_dir, exist_ok=True)
    columns = {
        'utt': utts,
        'audio_data': [open(utt2wav[u], 'rb').read() for u in utts],
        'wav': [utt2wav[u] for u in utts],
        'text': [utt2text[u] for u in utts],
        'spk': [utt2spk[u] for u in utts],
    }
    if instruct:
        columns['instruct'] = [instruct] * len(utts)
    parquet_file = os.path.join(parquet_dir, 'parquet_000000000.tar')
    pq.write_table(pa.table(columns), parquet_file)
    with open(os.path.join(parquet_dir, 'data.list'), 'w', encoding='utf-8') as f:
        f.write(parquet_file + '\n')
    return parquet_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_list', type=str,
                        default='/root/autodl-tmp/teochew-data/04_ChaoShan/train.list')
    parser.add_argument('--wav_dir', type=str,
                        default='/root/autodl-tmp/teochew-data/wav')
    parser.add_argument('--des_dir', type=str,
                        default='data')
    parser.add_argument('--instruct', type=str,
                        default='You are a helpful assistant. 请用潮汕话表达。<|endofprompt|>')
    parser.add_argument('--cv_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1986)
    args = parser.parse_args()

    utt2wav, utt2text, utt2spk, missing = parse_train_list(args.src_list, args.wav_dir)
    utts = sorted(utt2wav.keys())
    random.seed(args.seed)
    random.shuffle(utts)

    n_cv = max(1, int(len(utts) * args.cv_ratio))
    cv_utts = utts[:n_cv]
    train_utts = utts[n_cv:]

    print('[teochew prep] 有效 {} 句, 缺失音频 {} 句, 说话人 {}'.format(
        len(utts), missing, sorted(set(utt2spk.values()))))
    print('[teochew prep] train {} 句 / cv {} 句'.format(len(train_utts), len(cv_utts)))

    for name, sub in [('train', train_utts), ('cv', cv_utts)]:
        d = os.path.join(args.des_dir, name)
        os.makedirs(d, exist_ok=True)
        write_kaldi_files(d, sub, utt2wav, utt2text, utt2spk, args.instruct)
        pf = build_parquet(d, sub, utt2wav, utt2text, utt2spk, args.instruct)
        print('[teochew prep] {} -> {}'.format(name, pf))


if __name__ == '__main__':
    main()
