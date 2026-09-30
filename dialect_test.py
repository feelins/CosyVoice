#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文 zero-shot + 闽南/粤语方言 instruct 合成测试。"""
import os
import sys
import traceback

os.chdir('/root/autodl-tmp/CosyVoice')
sys.path.insert(0, 'third_party/Matcha-TTS')

try:
    import torch
    print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), flush=True)
    from cosyvoice.cli.cosyvoice import AutoModel
    import torchaudio

    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    print('模型加载完成，sample_rate =', cosyvoice.sample_rate, flush=True)

    prompt_wav = './asset/zero_shot_prompt.wav'

    # 1) 中文 zero-shot（验证管线）
    print('\n[1] 中文 zero-shot...', flush=True)
    for i, j in enumerate(cosyvoice.inference_zero_shot(
            '八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。',
            'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
            prompt_wav, stream=False, text_frontend=False)):
        torchaudio.save('zh_zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        print('  saved zh_zero_shot_{}.wav'.format(i), flush=True)

    # 2) 闽南话方言 instruct（潮汕的方向）
    print('\n[2] 闽南话方言 instruct...', flush=True)
    for i, j in enumerate(cosyvoice.inference_instruct2(
            '落雨天时，路顶湿湿，行人着细利。',
            'You are a helpful assistant. 请用闽南话表达。<|endofprompt|>',
            prompt_wav, stream=False)):
        torchaudio.save('minnan_instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        print('  saved minnan_instruct_{}.wav'.format(i), flush=True)

    # 3) 粤语方言 instruct（对照，README 示例里的）
    print('\n[3] 粤语方言 instruct...', flush=True)
    for i, j in enumerate(cosyvoice.inference_instruct2(
            '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。',
            'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
            prompt_wav, stream=False)):
        torchaudio.save('yue_instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        print('  saved yue_instruct_{}.wav'.format(i), flush=True)

    print('\nDONE', flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
