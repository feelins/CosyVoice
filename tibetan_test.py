#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""藏语 zero-shot 合成测试（Fun-CosyVoice3-0.5B）。"""
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

    # 藏语 prompt（参考音频 + 对应文本）
    prompt_text = 'བལ་ཡུལ་གྱི་ཕྱི་འབྲེལ་བློན་ཆེན་གྱིས་ཁས་ལེན་རྩ་བ་ནས་བྱེད་ཀྱི་མེད་པར།'
    prompt_wav = '/root/autodl-tmp/tibetan_wavs/wav/SP02-OTR002-01-A-0134.wav'
    # 要合成的藏语句子（另一句）
    tts_text = 'googleདྲ་ཚིགས་ལ་བརྟེན་ནས་སྐད་བསྒྱུར་བྱས་པའི་དབྱིན་ཇིའི་ཐོག་གི་ཁ་བརྡ་འདི་མཇུག་སྒྲིལ་འདོད་བྱུང་པ་དང་།'

    print('开始合成...', flush=True)
    # 藏语用 text_frontend=False：直接把原始藏文交给 Qwen tokenizer，避免中文前端（wetext）误处理
    for i, j in enumerate(cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False, text_frontend=False)):
        out = 'tibetan_zero_shot_{}.wav'.format(i)
        torchaudio.save(out, j['tts_speech'], cosyvoice.sample_rate)
        print('saved', out, flush=True)
    print('DONE', flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
