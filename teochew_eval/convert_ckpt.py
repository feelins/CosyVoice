#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 CosyVoice3 DeepSpeed 训练产出的 ZeRO 整体检查点，转成推理可用的标准 llm.pt，
并组装一个「微调版模型目录」，供 eval_teochew.py / webui.py / infer_teochew_real.py 直接加载。

为什么需要这一步:
  训练保存的是 DeepSpeed ZeRO 整体检查点(epoch_N_whole/ 目录, 内含 llm+flow+hift 全模型)。
  而推理端 CosyVoice3.load 要求 llm.pt 是「LLM 子模块」的 state_dict(strict=True),
  所以这里用 deepspeed 的 get_fp32_state_dict_from_zero_checkpoint 还原 fp32 权重,
  再抽取以 'llm.' 开头的键、去掉前缀, 得到纯 LLM 权重。

用法(训练结束后执行):
  cd /root/autodl-tmp/CosyVoice
  python teochew_eval/convert_ckpt.py \
      --exp_dir examples/teochew/exp/cosyvoice3/llm/torch_ddp \
      --pretrained pretrained_models/Fun-CosyVoice3-0.5B \
      --out pretrained_models/Fun-CosyVoice3-0.5B-teochew

说明:
  --exp_dir 必须是「含 latest 文件的父目录」(torch_ddp/), 脚本读 latest 自动选最新 epoch。
  --out 目录里除 llm.pt 外, 其余文件(flow.pt / hift.pt / *.onnx / cosyvoice3.yaml /
  CosyVoice-BlankEN / campplus.onnx 等)全部用软链接指向预训练目录, 几乎不占额外空间。
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, 'third_party/Matcha-TTS')

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def get_llm_expected_keys(pretrained):
    # 直接用预训练 llm.pt 的 key 集合作为 LLM 期望结构(与微调版架构一致, 无需重建模型)
    sd = torch.load(os.path.join(pretrained, 'llm.pt'),
                    map_location='cpu', weights_only=True)
    return set(sd.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', default='examples/teochew/exp/cosyvoice3/llm/torch_ddp')
    ap.add_argument('--pretrained', default='pretrained_models/Fun-CosyVoice3-0.5B')
    ap.add_argument('--out', default='pretrained_models/Fun-CosyVoice3-0.5B-teochew')
    args = ap.parse_args()

    # get_fp32_state_dict_from_zero_checkpoint 需要「含 latest 文件的父目录」,
    # 它会读 latest 自动定位到具体 epoch 的 tag 子目录。
    latest_file = os.path.join(args.exp_dir, 'latest')
    tag = open(latest_file).read().strip() if os.path.exists(latest_file) else '?'
    print('latest ->', tag)
    print('正在还原 fp32 权重(CPU, 可能要 1~2 分钟):', args.exp_dir)
    full_sd = get_fp32_state_dict_from_zero_checkpoint(args.exp_dir)

    # 自动探测 LLM 前缀
    prefixes = [p for p in ('llm.', 'module.llm.')
                if any(k.startswith(p) for k in full_sd.keys())]
    assert prefixes, '检查点里找不到 llm. 前缀的权重, 结构可能不对'
    prefix = prefixes[0]
    print('LLM 权重前缀:', prefix)

    expected = get_llm_expected_keys(args.pretrained)
    # 预训练 llm.pt 的 key 既包含 'llm.' 前缀的子模块, 也包含顶层的
    # llm_decoder.weight / speech_embedding.weight, 故直接取「全模型键 ∩ 预训练键」最稳妥。
    llm_sd = {k: v for k, v in full_sd.items() if k in expected}
    missing = expected - set(llm_sd.keys())
    extra = set(llm_sd.keys()) - expected
    assert not missing, 'LLM 权重缺失: {}'.format(sorted(missing)[:10])
    assert not extra, 'LLM 有多余键: {}'.format(sorted(extra)[:10])
    print('LLM 权重对齐 OK, 共 {} 个参数'.format(len(llm_sd)))

    # 组装 out 目录(非 llm 文件全部软链)
    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.pretrained):
        src = os.path.abspath(os.path.join(args.pretrained, name))
        dst = os.path.join(args.out, name)
        if name in ('llm.pt', 'llm.rl.pt'):
            continue  # 这两个我们用微调版替换
        if os.path.islink(dst) or os.path.exists(dst):
            continue
        os.symlink(src, dst)
    torch.save(llm_sd, os.path.join(args.out, 'llm.pt'))
    print('已写出:', os.path.join(args.out, 'llm.pt'))
    print('完成! 随后可执行:')
    print('  python teochew_eval/infer_teochew_real.py --model_dir {}'.format(args.out))


if __name__ == '__main__':
    main()
