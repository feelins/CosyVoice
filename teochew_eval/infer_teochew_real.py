#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从训练集 parquet 抽取真实潮汕话句子 + 原生说话人音色，做推理合成。
同时支持 --compare 在原版 / 微调版上各合成一遍，便于 A/B 试听。

数据来源: examples/teochew/data/train/parquet/parquet_000000000.tar
  字段: text(潮汕话句子) / instruct(请用潮汕话表达。) / audio_data(原生说话人 wav 字节)

用法:
  cd /root/autodl-tmp/CosyVoice
  # 只生成微调版
  python teochew_eval/infer_teochew_real.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B-teochew
  # 原版 vs 微调版 A/B
  python teochew_eval/infer_teochew_real.py --compare
"""
import os
import sys
import argparse

ROOT = '/root/autodl-tmp/CosyVoice'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party/Matcha-TTS'))

import torch
import torchaudio
import pyarrow.parquet as pq
from cosyvoice.cli.cosyvoice import AutoModel

PARQUET = 'examples/teochew/data/train/parquet/parquet_000000000.tar'
# 注意: instruct 必须带 <|endofprompt|> 标记(CosyVoice3 的硬性要求), 否则推理断言失败
INSTRUCT = 'You are a helpful assistant. 请用潮汕话表达。<|endofprompt|>'
PROMPT_PATH = 'teochew_eval/prompt_chaoshan.wav'


def pick_samples(n_text=8):
    t = pq.read_table(PARQUET)
    texts = t.column('text').to_pylist()
    audios = t.column('audio_data').to_pylist()
    # 过滤过长/过短, 并剔除含罕见字符(>U+FFFF, 如 𠁞)的句子, 避免分词后输出过短打不过 vocoder
    cand = [(tx, au) for tx, au in zip(texts, audios)
            if 4 <= len(str(tx)) <= 36
            and all(ord(c) < 0xFFFF for c in str(tx))]
    step = max(1, len(cand) // (n_text + 1))
    sel = cand[::step][: n_text + 1]
    prompt_audio = sel[0][1]          # 原生说话人音色
    target_texts = [s[0] for s in sel[1:n_text + 1]]
    return prompt_audio, target_texts


def synth(cosyvoice, out_dir, prefix, prompt_path, target_texts):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, tx in enumerate(target_texts):
        try:
            # text_frontend=False: 潮汕话/方言文本不做中文归一化, 否则可能产出异常短的输出
            for out in cosyvoice.inference_instruct2(
                    tx, INSTRUCT, prompt_path, stream=False, text_frontend=False):
                p = os.path.join(out_dir, '{}_teochew_{}.wav'.format(prefix, i))
                torchaudio.save(p, out['tts_speech'], cosyvoice.sample_rate)
                print('saved', p, flush=True)
                manifest.append((p, tx))
        except Exception as e:
            print('!! 第 {} 句合成失败: {} | 文本: {}'.format(i, repr(e), tx), flush=True)
    # 保存文本清单，方便对照试听
    with open(os.path.join(out_dir, 'manifest.txt'), 'w', encoding='utf-8') as f:
        for p, tx in manifest:
            f.write('{}\t{}\n'.format(os.path.basename(p), tx))
    print('清单已写', os.path.join(out_dir, 'manifest.txt'), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir',
                    default='pretrained_models/Fun-CosyVoice3-0.5B-teochew')
    ap.add_argument('--out', default='teochew_eval/out')
    ap.add_argument('--n_text', type=int, default=8)
    ap.add_argument('--compare', action='store_true')
    args = ap.parse_args()

    prompt_audio, target_texts = pick_samples(args.n_text)
    with open(PROMPT_PATH, 'wb') as f:
        f.write(prompt_audio)
    print('参考音色:', PROMPT_PATH, '| 目标潮汕话句数:', len(target_texts), flush=True)
    for tx in target_texts:
        print('  -', tx, flush=True)

    if args.compare:
        print('加载原版...', flush=True)
        synth(AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B'),
              os.path.join(args.out, 'pretrained'), 'base',
              PROMPT_PATH, target_texts)
        print('加载微调版...', flush=True)
        synth(AutoModel(model_dir=args.model_dir),
              os.path.join(args.out, 'finetuned'), 'ft',
              PROMPT_PATH, target_texts)
    else:
        print('加载微调版:', args.model_dir, flush=True)
        synth(AutoModel(model_dir=args.model_dir),
              args.out, 'ft', PROMPT_PATH, target_texts)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
