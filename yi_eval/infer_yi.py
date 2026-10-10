#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""彝语三方对照推理: origin(原始录音) / pretrain(底座 Fun-CosyVoice3-0.5B) / finetune(当前微调版)。

文本传 X-SAMPA(纯 ASCII, 训练时即为此表示), text_frontend=False,
instruct 用「请用彝语表达。」(CosyVoice3 硬性要求带 <|endofprompt|> 标记)。

参考音色(prompt)取自真实彝语(06)说话人 wav, 目标句用其余样本;
origin 为各目标句对应的原始录音(ground truth), 便于与合成结果对照。

用法:
  cd /root/autodl-tmp/CosyVoice
  python yi_eval/infer_yi.py --tsv yi_eval/samples.tsv --compare --origin
  python yi_eval/infer_yi.py --tsv yi_eval/samples.tsv --origin          # 仅微调版
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

INSTRUCT = 'You are a helpful assistant. 请用彝语表达。<|endofprompt|>'


def load_pairs(tsv):
    pairs = []
    with open(tsv, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            _, wav, text = parts[0], parts[1], parts[2]
            if os.path.exists(wav):
                pairs.append((wav, text))
            else:
                print('!! 缺失 wav:', wav, flush=True)
    return pairs


def synth(cosyvoice, out_dir, prefix, prompt_wav, target_texts):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i, tx in enumerate(target_texts):
        try:
            for out in cosyvoice.inference_instruct2(
                    tx, INSTRUCT, prompt_wav, stream=False, text_frontend=False):
                p = os.path.join(out_dir, '{}_yi_{}.wav'.format(prefix, i))
                torchaudio.save(p, out['tts_speech'], cosyvoice.sample_rate)
                print('saved', p, flush=True)
                manifest.append((p, tx))
        except Exception as e:
            print('!! 第 {} 句合成失败: {} | {}'.format(i, repr(e), tx), flush=True)
    with open(os.path.join(out_dir, 'manifest.txt'), 'w', encoding='utf-8') as f:
        for p, tx in manifest:
            f.write('{}\t{}\n'.format(os.path.basename(p), tx))
    print('清单:', os.path.join(out_dir, 'manifest.txt'), flush=True)


def copy_origin(target_pairs, out_dir):
    od = os.path.join(out_dir, 'orig')
    os.makedirs(od, exist_ok=True)
    for i, (wav, _tx) in enumerate(target_pairs):
        dst = os.path.join(od, 'orig_yi_{}.wav'.format(i))
        if not os.path.exists(dst):
            shutil.copy(wav, dst)
        print('orig', dst, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B-yi')
    ap.add_argument('--base_model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    ap.add_argument('--out', default='yi_eval/out')
    ap.add_argument('--tsv', default='yi_eval/samples.tsv')
    ap.add_argument('--n_text', type=int, default=5)
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--origin', action='store_true')
    args = ap.parse_args()

    pairs = load_pairs(args.tsv)
    assert pairs, 'tsv 无可用 (wav, xsampa) 配对'
    prompt_wav, prompt_text = pairs[0]
    target_pairs = pairs[1:args.n_text + 1]
    target_texts = [p[1] for p in target_pairs]
    print('参考音色:', prompt_wav, flush=True)
    print('prompt文本(xsampa):', prompt_text, flush=True)
    for i, tx in enumerate(target_texts):
        print('  目标{}: {}'.format(i, tx), flush=True)

    if args.origin:
        copy_origin(target_pairs, args.out)
    if args.compare:
        print('加载底座...', flush=True)
        synth(AutoModel(model_dir=args.base_model_dir),
              os.path.join(args.out, 'pretrained'), 'base', prompt_wav, target_texts)
        print('加载微调版...', flush=True)
        synth(AutoModel(model_dir=args.model_dir),
              os.path.join(args.out, 'finetuned'), 'ft', prompt_wav, target_texts)
    else:
        print('加载微调版:', args.model_dir, flush=True)
        synth(AutoModel(model_dir=args.model_dir),
              args.out, 'ft', prompt_wav, target_texts)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
