#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 CosyVoice3 DeepSpeed 训练产出的 epoch_N_whole/mp_rank_00_model_states.pt
(即完整模型权重, 不是 ZeRO 优化器分片) 转成推理可用的独立 llm.pt, 并组装一个
「微调版模型目录」(其余文件软链到预训练目录, 几乎不占空间), 供 infer_yi.py 直接加载。

与 teochew_eval/convert_ckpt.py 的区别: 直接读 model_states.pt, 不依赖 latest 文件
(本实验中 latest 指向已被 watch_disk 删掉的 epoch_13_whole) 也不依赖已损坏的优化器
状态文件, 因此即使在「存盘中断 + 旧检查点被清」的残局下也能还原权重。

用法:
  cd /root/autodl-tmp/CosyVoice
  python yi_eval/convert_yi_ckpt.py \
      --ckpt yi-corpus/exp/yi/llm/torch_ddp/epoch_14_whole/mp_rank_00_model_states.pt \
      --out pretrained_models/yi-ft-ep14
"""
import os
import argparse
import torch

ROOT = '/root/autodl-tmp/CosyVoice'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True,
                    help='epoch_N_whole/mp_rank_00_model_states.pt')
    ap.add_argument('--pretrained', default='pretrained_models/Fun-CosyVoice3-0.5B')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # 1) 预训练 llm.pt 的 key 集合 = LLM 期望结构
    base_pt = os.path.join(args.pretrained, 'llm.pt')
    base_sd = torch.load(base_pt, map_location='cpu', weights_only=True)
    expected = set(base_sd.keys())
    print('base llm.pt key 数:', len(expected))
    print('base key 样例:', sorted(expected)[:3])

    # 2) 读训练检查点的模型权重(完整 zip, 已校验有效)
    #    CosyVoice3 的 model_states.pt 顶层含大量训练元数据, 真实模型权重在
    #    ckpt['module'] 下(键形如 llm.model...), 与 base llm.pt 的键一一对应。
    ckpt = torch.load(args.ckpt, map_location='cpu')
    inner = ckpt.get('module', ckpt.get('mp_rank_00_model_states', ckpt))
    print('ckpt 内层 key 样例:', sorted(inner.keys())[:3])

    # 3) 抽 LLM 权重: 内层 key 与 base key 求交集(自动试带/不带 llm. 前缀)
    def build(strip_prefix):
        sd = {}
        for k, v in inner.items():
            sk = k[len(strip_prefix):] if (strip_prefix and k.startswith(strip_prefix)) else k
            if sk in expected:
                sd[sk] = v
        return sd

    llm_sd = None
    for pref in ('llm.', ''):
        sd = build(pref)
        if set(sd.keys()) == expected:
            print('LLM 权重对齐 OK (前缀=%r), 共 %d 个参数' % (pref, len(sd)))
            llm_sd = sd
            break
    if llm_sd is None:
        got = set(build('').keys())
        missing = sorted(expected - got)[:8]
        extra = sorted(got - expected)[:8]
        raise SystemExit('key 不匹配 missing=%s extra=%s' % (missing, extra))

    # 4) 组装 out 目录: 非 llm 文件全部软链到预训练目录
    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.pretrained):
        src = os.path.abspath(os.path.join(args.pretrained, name))
        dst = os.path.join(args.out, name)
        if name in ('llm.pt', 'llm.rl.pt'):
            continue  # 用微调版替换
        if os.path.islink(dst) or os.path.exists(dst):
            continue
        os.symlink(src, dst)
    out_llm = os.path.join(args.out, 'llm.pt')
    torch.save(llm_sd, out_llm)
    print('已写出:', out_llm)
    print('完成! 随后可执行:')
    print('  python yi_eval/infer_yi.py --model_dir %s --tsv yi_eval/samples.tsv --compare --origin' % args.out)


if __name__ == '__main__':
    main()
