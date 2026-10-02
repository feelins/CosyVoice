#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""藏语零样本复刻推理: 基于 CosyVoice3 微调版(epoch_17)合成藏语(Wylie 罗马化文本)。

数据/原理:
  - 训练文本用的是 Wylie 罗马化(底座分词器对藏文 Unicode 覆盖 0/13, 对 Wylie 覆盖 12/13),
    所以推理 text 也必须传 Wylie, 且 text_frontend=False(不做中文归一化, 否则输出异常)。
  - instruct 沿用训练时的「请用藏语表达。」, 并带 CosyVoice3 硬性要求的 <|endofprompt|> 标记。
  - 参考音色(prompt)取自真实藏语说话人 wav, prompt 文本用对应的 Wylie 转写。

用法:
  cd /root/autodl-tmp/CosyVoice
  python tibetan_eval/infer_tibetan.py                       # 仅微调版
  python tibetan_eval/infer_tibetan.py --compare             # 原版 vs 微调版 A/B
  python tibetan_eval/infer_tibetan.py --n_text 6 --out tibetan_eval/out
"""
import os
import sys
import shutil
import argparse

ROOT = '/root/autodl-tmp/CosyVoice'
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'third_party/Matcha-TTS'))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

WYLVIE_TSV = '/root/autodl-tmp/test-wylie.tsv'      # 列: idx \t path \t wylie句
WAV_DIR = '/root/autodl-tmp/tibetan_wavs/wav'        # 真实藏语说话人 wav
INSTRUCT = 'You are a helpful assistant. 请用藏语表达。<|endofprompt|>'


def load_pairs(tsv, speaker=None):
    pairs = []
    with open(tsv, encoding='utf-8') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            _, path, wylie = parts[0], parts[1], parts[2]
            if speaker and speaker not in path:
                continue
            wav = os.path.join(WAV_DIR, path)
            if os.path.exists(wav):
                pairs.append((wav, wylie))
    return pairs


def synth(cosyvoice, out_dir, prefix, prompt_wav, target_texts):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, tx in enumerate(target_texts):
        try:
            for out in cosyvoice.inference_instruct2(
                    tx, INSTRUCT, prompt_wav, stream=False, text_frontend=False):
                p = os.path.join(out_dir, '{}_tibetan_{}.wav'.format(prefix, i))
                torchaudio.save(p, out['tts_speech'], cosyvoice.sample_rate)
                print('saved', p, flush=True)
                manifest.append((p, tx))
        except Exception as e:
            print('!! 第 {} 句合成失败: {} | 文本: {}'.format(i, repr(e), tx), flush=True)
    with open(os.path.join(out_dir, 'manifest.txt'), 'w', encoding='utf-8') as f:
        for p, tx in manifest:
            f.write('{}\t{}\n'.format(os.path.basename(p), tx))
    print('清单已写', os.path.join(out_dir, 'manifest.txt'), flush=True)


def copy_origin(target_pairs, out_dir):
    """把目标句对应的原始录音(ground truth)拷到 out/orig/, 便于三方对照。"""
    od = os.path.join(out_dir, 'orig')
    os.makedirs(od, exist_ok=True)
    for i, (wav, _tx) in enumerate(target_pairs):
        dst = os.path.join(od, 'orig_tibetan_{}.wav'.format(i))
        if not os.path.exists(dst):
            shutil.copy(wav, dst)
        print('orig', dst, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir',
                    default='pretrained_models/Fun-CosyVoice3-0.5B-tibetan')
    ap.add_argument('--base_model_dir',
                    default='pretrained_models/Fun-CosyVoice3-0.5B')
    ap.add_argument('--out', default='tibetan_eval/out')
    ap.add_argument('--n_text', type=int, default=5)
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--origin', action='store_true',
                    help='同时导出目标句对应的原始录音(ground truth)做对照')
    ap.add_argument('--tsv', default=WYLVIE_TSV,
                    help='wylie tsv 路径(test 集默认), 如换 train-wylie.tsv 取更多女声')
    ap.add_argument('--speaker', default=None,
                    help='按文件名子串筛选说话人, 如 OTR002-01(女声) / OTR002-02(男声)')
    args = ap.parse_args()

    pairs = load_pairs(args.tsv, args.speaker)
    assert pairs, '未找到可用的 (wav, wylie) 配对'
    prompt_wav, prompt_text = pairs[0]
    target_pairs = pairs[1:args.n_text + 1]
    target_texts = [p[1] for p in target_pairs]
    print('参考音色:', prompt_wav, '| prompt文本(wylie):', prompt_text, flush=True)
    for i, tx in enumerate(target_texts):
        print('  目标{}: {}'.format(i, tx), flush=True)

    if args.origin:
        copy_origin(target_pairs, args.out)
    if args.compare:
        print('加载原版...', flush=True)
        synth(AutoModel(model_dir=args.base_model_dir),
              os.path.join(args.out, 'pretrained'), 'base',
              prompt_wav, target_texts)
        print('加载微调版...', flush=True)
        synth(AutoModel(model_dir=args.model_dir),
              os.path.join(args.out, 'finetuned'), 'ft',
              prompt_wav, target_texts)
    else:
        print('加载微调版:', args.model_dir, flush=True)
        synth(AutoModel(model_dir=args.model_dir),
              args.out, 'ft', prompt_wav, target_texts)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
