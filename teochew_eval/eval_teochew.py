#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""潮汕话(CosyVoice3 微调)效果验证: 合成样本 + (可选)与原版 A/B 对比。主观试听为主。

用法:
  cd /root/autodl-tmp/CosyVoice
  python teochew_eval/eval_teochew.py --compare --out teochew_eval/out
  python teochew_eval/eval_teochew.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B-teochew

注意:
  - 句子在 TEochew_INSTRUCT 里改(换成你自己的真实潮汕话语料)。
  - instruct 必须含 <|endofprompt|>; 方言文本要 text_frontend=False。
  - --prompt 默认 asset/zero_shot_prompt.wav(通用音色), 想验证潮汕音色请换成本地潮汕人声。
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
from cosyvoice.cli.cosyvoice import AutoModel

DEFAULT_PROMPT = './asset/zero_shot_prompt.wav'
# instruct 必须带 <|endofprompt|>, 否则推理断言失败
INSTRUCT = 'You are a helpful assistant. 请用潮汕话表达。<|endofprompt|>'

# ===== 把下面的句子换成你自己的潮汕话语料 =====
TEochew_INSTRUCT = [
    '落雨哩，路咧澹澹，行路爱细腻。',
    '伊昨昏去街路买一尾鱼转来。',
    '恁食饱未？行出去行行仔。',
    '这领衫诚雅，偌多钱？',
]
# 中文 zero-shot：验证管线 & 音色克隆（prompt_text 需与参考音频转写一致）
ZH_ZEROSHOT = [
    ('欢迎使用潮汕话语音合成，这是一段测试语音。',
     '希望你以后能够做的比我还好呦。'),
]


def synth_one(cosyvoice, out_dir, prefix, prompt_wav):
    os.makedirs(out_dir, exist_ok=True)
    for i, txt in enumerate(TEochew_INSTRUCT):
        for out in cosyvoice.inference_instruct2(
                txt, INSTRUCT, prompt_wav, stream=False, text_frontend=False):
            p = os.path.join(out_dir, '{}_teochew_{}.wav'.format(prefix, i))
            torchaudio.save(p, out['tts_speech'], cosyvoice.sample_rate)
            print('saved', p, flush=True)
    for i, (txt, prompt_text) in enumerate(ZH_ZEROSHOT):
        for out in cosyvoice.inference_zero_shot(
                txt, prompt_text, prompt_wav, stream=False, text_frontend=False):
            p = os.path.join(out_dir, '{}_zh_{}.wav'.format(prefix, i))
            torchaudio.save(p, out['tts_speech'], cosyvoice.sample_rate)
            print('saved', p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', default='pretrained_models/Fun-CosyVoice3-0.5B')
    ap.add_argument('--out', default='teochew_eval/out')
    ap.add_argument('--prompt', default=DEFAULT_PROMPT)
    ap.add_argument('--compare', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.prompt):
        print('!! 参考音频不存在:', args.prompt)
        sys.exit(1)

    if args.compare:
        print('加载原版...', flush=True)
        m0 = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
        print('加载微调版...', flush=True)
        m1 = AutoModel(model_dir=args.model_dir)
        synth_one(m0, os.path.join(args.out, 'pretrained'), 'base', args.prompt)
        synth_one(m1, os.path.join(args.out, 'finetuned'), 'ft', args.prompt)
    else:
        print('加载模型:', args.model_dir, flush=True)
        m = AutoModel(model_dir=args.model_dir)
        synth_one(m, args.out, 'ft', args.prompt)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
