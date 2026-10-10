#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练期磁盘守卫：只保留最近 K 个 epoch_*_whole 检查点，避免像藏语那样磁盘写满崩溃。

用法:
  python3 scripts/watch_disk.py &      # 后台运行，随训练一同启动
默认监控 exp/yi/llm/torch_ddp，每 30s 清理一次，保留最新 3 个 epoch 检查点；
若剩余磁盘 < 8GB，则激进到只保留最新 1 个。日志写到 yi-corpus/cleanup.log。
"""
import os
import sys
import time
import shutil
import logging

CKPT_DIR = '/root/autodl-tmp/CosyVoice/yi-corpus/exp/yi/llm/torch_ddp'
KEEP = 3
DISK_PATH = '/root/autodl-tmp'
LOW_DISK_GB = 8.0
INTERVAL = 30

logging.basicConfig(
    filename='/root/autodl-tmp/CosyVoice/yi-corpus/cleanup.log',
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def free_gb():
    st = os.statvfs(DISK_PATH)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def cleanup():
    if not os.path.isdir(CKPT_DIR):
        return
    dirs = [d for d in os.listdir(CKPT_DIR)
            if d.startswith('epoch_') and d.endswith('_whole')
            and os.path.isdir(os.path.join(CKPT_DIR, d))]
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(CKPT_DIR, d)), reverse=True)
    keep = KEEP
    fg = free_gb()
    if fg < LOW_DISK_GB:
        keep = 1
        logging.warning('剩余磁盘 %.1fG < %.1fG，激进保留最新 1 个检查点', fg, LOW_DISK_GB)
    for d in dirs[keep:]:
        p = os.path.join(CKPT_DIR, d)
        try:
            shutil.rmtree(p)
            logging.info('删除旧检查点 %s（保留最新 %d 个）', d, keep)
        except Exception as e:
            logging.error('删除 %s 失败: %s', d, e)


def main():
    logging.info('watch_disk 启动，监控 %s，保留最新 %d 个 epoch 检查点', CKPT_DIR, KEEP)
    while True:
        try:
            cleanup()
        except Exception as e:
            logging.error('cleanup 异常: %s', e)
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
