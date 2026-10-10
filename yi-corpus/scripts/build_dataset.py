#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""彝语语料 -> 统一 X-SAMPA 训练文本（中间结果）。

读取 yi-corpus/metadata.csv（含 BOM，用 utf-8-sig）+ metadata.jsonl，
按来源分别转换：
  - 06-chuxiong-yi-meige: text 已是 IPA -> ipa2xsampa -> X-SAMPA
  - 04-iiiwav: text 是凉山拼音 -> roman2ipa -> IPA -> ipa2xsampa -> X-SAMPA

产出（yi-corpus/processed/）:
  all_xsampa.tsv / all_xsampa.jsonl   全量（含 text_ipa / text_roman / use_for_train）
  stats.json                         统计与告警
  splits/train.tsv, valid.tsv, test.tsv            06+04 分层切分（prepare_parquet 用）
  splits/train_06.tsv, valid_06.tsv, test_06.tsv   仅 06（推荐首训）

切分 tsv 格式:  idx\t<wav相对路径>\t<xsampa>   （prepare_parquet 读取）

说明: use_for_train 默认 True；超长(>60s)/超短(<1s) 音频由训练配置 filter
自动丢弃，这里只在 stats 里标注，不预先删除（保留全部中间结果）。
"""
import csv
import json
import os
import sys
import collections
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipa2xsampa import convert as ipa2xsampa
from roman2ipa import convert_sentence as roman2ipa

ROOT = '/root/autodl-tmp/CosyVoice/yi-corpus'
OUT = os.path.join(ROOT, 'processed')
SPLITS = os.path.join(OUT, 'splits')
os.makedirs(SPLITS, exist_ok=True)


def load_rows():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'metadata.csv'), encoding='utf-8-sig')))
    rom = {}
    jl = os.path.join(ROOT, 'metadata.jsonl')
    if os.path.exists(jl):
        for line in open(jl, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rom[d.get('id')] = d
    return rows, rom


def to_xsampa(row, rom):
    src = row['source']
    text = (row.get('text') or '').strip()
    jd = rom.get(row['id'], {})
    if jd.get('text_ipa'):
        ipa = jd['text_ipa'].strip()
        roman = jd.get('text_roman') or text
    elif src == '06-chuxiong-yi-meige':
        ipa = text
        roman = ''
    else:
        roman = text
        ipa = roman2ipa(roman)
    xs = ipa2xsampa(ipa)
    return ipa, roman, xs


def split(rec_list, tag):
    rec_list = sorted(rec_list, key=lambda x: x['id'])
    n = len(rec_list)
    nv = max(30, n // 20)
    nt = max(30, n // 20)
    test = rec_list[:nt]
    valid = rec_list[nt:nt + nv]
    train = rec_list[nt + nv:]
    for name, sub in [('train', train), ('valid', valid), ('test', test)]:
        with open(os.path.join(SPLITS, f'{name}{tag}.tsv'), 'w', encoding='utf-8') as f:
            f.write('idx\twav\txsampa\n')
            for x in sub:
                f.write(f"{x['id']}\t{x['wav']}\t{x['text_xsampa']}\n")
    return len(train), len(valid), len(test)


def main():
    rows, rom = load_rows()
    recs = []
    stats = {'n': len(rows), 'by_source': collections.Counter(),
             'dur': {}, 'xsampa_empty': 0, 'warnings': []}
    for r in rows:
        src = r['source']
        stats['by_source'][src] += 1
        ipa, roman, xs = to_xsampa(r, rom)
        try:
            dur = float(r['duration'])
        except Exception:
            dur = 0.0
        recs.append({
            'id': r['id'], 'source': src, 'wav': r['wav'], 'duration': round(dur, 3),
            'text_xsampa': xs, 'text_ipa': ipa, 'text_roman': roman,
            'text_zh': r.get('text_zh', ''), 'speaker_note': r.get('speaker_note', ''),
            'use_for_train': True,
        })
        if not xs.strip():
            stats['xsampa_empty'] += 1

    for src in stats['by_source']:
        ds = [x['duration'] for x in recs if x['source'] == src]
        stats['dur'][src] = {'n': len(ds), 'sum_h': round(sum(ds) / 3600, 3),
                             'min': round(min(ds), 3), 'med': round(statistics.median(ds), 3),
                             'max': round(max(ds), 3), 'over60': sum(d > 60 for d in ds),
                             'under1': sum(d < 1 for d in ds)}

    with open(os.path.join(OUT, 'all_xsampa.tsv'), 'w', encoding='utf-8') as f:
        f.write('id\tsource\twav\tduration\ttext_xsampa\ttext_ipa\ttext_roman\tuse_for_train\n')
        for x in recs:
            f.write('\t'.join(str(x[k]) for k in
                    ['id', 'source', 'wav', 'duration', 'text_xsampa', 'text_ipa', 'text_roman', 'use_for_train']) + '\n')
    with open(os.path.join(OUT, 'all_xsampa.jsonl'), 'w', encoding='utf-8') as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')

    a = split([x for x in recs], '')
    b = split([x for x in recs if x['source'] == '06-chuxiong-yi-meige'], '_06')
    stats['splits_all'] = a
    stats['splits_06'] = b
    with open(os.path.join(OUT, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('[build_dataset] 全量:', len(recs), ' xsampa空:', stats['xsampa_empty'])
    print('[build_dataset] splits(all) train/valid/test =', a)
    print('[build_dataset] splits(06)  train/valid/test =', b)
    print('[build_dataset] 写出目录:', OUT)


if __name__ == '__main__':
    main()
