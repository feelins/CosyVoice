#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""抓取 CosyVoice3 的中间 token：文本 token（输入）+ 语音 token（LLM 输出）。"""
import os
import sys
import traceback

os.chdir('/root/autodl-tmp/CosyVoice')
sys.path.insert(0, 'third_party/Matcha-TTS')

try:
    import torch
    from transformers import AutoTokenizer
    from cosyvoice.cli.cosyvoice import AutoModel

    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    f = cosyvoice.frontend
    m = cosyvoice.model

    # 直接用原始 Qwen tokenizer 解码
    raw_tok = AutoTokenizer.from_pretrained('pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN')

    tts_text = '落雨天时，路顶湿湿，行人着细利。'
    instruct_text = 'You are a helpful assistant. 请用闽南话表达。<|endofprompt|>'
    prompt_wav = './asset/zero_shot_prompt.wav'

    # ---- 1) 文本 token（输入侧）----
    text_token, text_len = f._extract_text_token(tts_text)
    prompt_token, prompt_len = f._extract_text_token(instruct_text)
    print('=' * 60)
    print('[文本 token]')
    print('  tts_text 原文   :', tts_text)
    print('  tts_text token数:', text_token.shape[1])
    print('  tts_text decode :', repr(raw_tok.decode(text_token[0].tolist())))
    print('  instruct 原文   :', instruct_text)
    print('  instruct token数:', prompt_token.shape[1])
    print('  instruct decode :', repr(raw_tok.decode(prompt_token[0].tolist())))

    # ---- 2) 语音 token（LLM 输出，即中间序列）----
    model_input = f.frontend_instruct2(tts_text, instruct_text, prompt_wav, cosyvoice.sample_rate, '')
    with torch.inference_mode():
        gen = m.llm.inference(
            text=model_input['text'].to(m.device),
            text_len=model_input['text_len'].to(m.device),
            prompt_text=model_input['prompt_text'].to(m.device),
            prompt_text_len=model_input['prompt_text_len'].to(m.device),
            prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32).to(m.device),
            prompt_speech_token_len=torch.tensor([0], dtype=torch.int32).to(m.device),
            embedding=model_input['llm_embedding'].to(m.device),
        )
        speech_tokens = [int(i) for i in gen]

    print('=' * 60)
    print('[语音 token（LLM 输出，25 token/秒）]')
    print('  speech token 数:', len(speech_tokens))
    print('  对应时长(约)  :', round(len(speech_tokens) / 25, 2), '秒')
    print('  前 40 个值     :', speech_tokens[:40])
    print('  值域(最小~最大):', min(speech_tokens), '~', max(speech_tokens))
    print('=' * 60)
    print('注：语音 token 是 6561 码表的离散码，不是 IPA/音素，')
    print('    无法直接还原成可读音素序列，但可以看数量/分布来对照内容长短。')
except Exception:
    traceback.print_exc()
    sys.exit(1)
