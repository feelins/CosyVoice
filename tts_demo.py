#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CosyVoice3 合成模板 —— 只改下面【要改的地方】即可。"""
import os
import sys

os.chdir('/root/autodl-tmp/CosyVoice')
sys.path.insert(0, 'third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# ================= 要改的地方 =================
TEXT = '落雨天时，路顶湿湿，行人着细利。'          # ① 要合成的内容（汉字）
PROMPT_WAV = './asset/zero_shot_prompt.wav'       # ② 参考音频（决定音色/音色克隆源）
MODE = 'instruct2'                                 # ③ 'zero_shot' 或 'instruct2'
INSTRUCT = 'You are a helpful assistant. 请用闽南话表达。<|endofprompt|>'  # ④ 方言指令
# 方言指令可换成：请用广东话表达 / 请用四川话表达 / 请用东北话表达 / 请用普通话表达 ...
# =============================================

cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

if MODE == 'zero_shot':
    # zero-shot：克隆参考音频的音色，按普通话读 TEXT
    prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
    for i, j in enumerate(cosyvoice.inference_zero_shot(TEXT, prompt_text, PROMPT_WAV, stream=False, text_frontend=False)):
        torchaudio.save('out_zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        print('saved out_zero_shot_{}.wav'.format(i))
elif MODE == 'instruct2':
    # instruct2：按 INSTRUCT 里的指令（方言/情绪/语速）读 TEXT
    for i, j in enumerate(cosyvoice.inference_instruct2(TEXT, INSTRUCT, PROMPT_WAV, stream=False)):
        torchaudio.save('out_instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        print('saved out_instruct_{}.wav'.format(i))

print('DONE')
