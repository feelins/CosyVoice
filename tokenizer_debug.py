#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 Qwen tokenizer 对藏文的处理。"""
from transformers import AutoTokenizer
import sys
sys.path.insert(0, '/root/autodl-tmp/CosyVoice')

tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN')
print('vocab_size =', tok.vocab_size)

text = 'བལ་ཡུལ་གྱི་ཕྱི་འབྲེལ་བློན་ཆེན་གྱིས་ཁས་ལེན་རྩ་བ་ནས་བྱེད་ཀྱི་མེད་པར།'
ids = tok.encode(text)
print('藏文 token 数 =', len(ids))
print('前 30 个 token =', tok.convert_ids_to_tokens(ids[:30]))
# 反解回文本
print('decode =', repr(tok.decode(ids)))

zh = '你好，世界。'
ids2 = tok.encode(zh)
print('\n中文 token 数 =', len(ids2))
print('中文 tokens =', tok.convert_ids_to_tokens(ids2))
