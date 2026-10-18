"""Create an independent, byte-preserving copy of the two completed grids."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import time

R = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
DEST = R / 'rollout_archive/grid_20260912_20260913'
STAGE = DEST.with_name(DEST.name + '.building')
SOURCES = {
    'math': (R / 'opd_eval_hard/nojudge_grid_20260912', {'math500': 500, 'olympiad_bench': 674, 'bbh': 1080}, 25),
    'code': (R / 'opd_eval_code/code_reasoning_grid_20260913', {'cruxeval_i': 800, 'cruxeval_o': 800, 'humaneval_plus': 164, 'mbpp_plus': 378, 'livecodebench': 175}, 27),
}

assert not DEST.exists(), f'Destination already exists: {DEST}'
assert not (STAGE/'COMPLETE.json').exists(), 'Completed staging must not be overwritten'
STAGE.mkdir(parents=True, exist_ok=True)
manifest = dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), destination=str(DEST),
                storage='independent regular-file copies; no symlinks or hard links',
                raw_json_modified=False, sources={}, groups=[], total_rows=0, total_raw_bytes=0)
fields = ['category', 'dataset', 'config', 'source_config', 'k', 'threshold', 'problem_id',
          'dataset_index', 'finish_reason', 'response_tokens', 'relative_path', 'source_path', 'sha256', 'bytes']
all_seen = set()
with (STAGE / 'index.csv').open('w', newline='') as index:
    writer = csv.DictWriter(index, fieldnames=fields)
    writer.writeheader()
    for category, (root, expected, expected_groups) in SOURCES.items():
        configs = json.loads((root / 'configs.json').read_text())
        assert len(configs) == expected_groups
        assert len({c['name'] for c in configs}) == expected_groups
        prompts = [json.loads(line) for line in (root/'prompts.jsonl').read_text().splitlines()]
        prompt_ids = {r['dataset_index']: (r['dataset'], r['problem_id']) for r in prompts}
        assert len(prompt_ids) == len(prompts) == sum(expected.values())
        manifest['sources'][category] = str(root)
        meta = STAGE / 'metadata' / category
        meta.mkdir(parents=True, exist_ok=True)
        for name in ['configs.json', 'prompts.jsonl', 'data_manifest.json', 'ALL_GENERATION_COMPLETE.json']:
            source = root / name
            if source.is_file():
                (meta / name).write_bytes(source.read_bytes())
        for config in configs:
            source_config = config['name']
            canonical = f"k{config['k']}_t{config['threshold']:g}" if 'k' in config else source_config
            files = sorted((root / source_config / 'completed').glob('*.json'))
            assert len(files) == sum(expected.values()), (category, source_config, len(files))
            counts = collections.Counter()
            finishes = collections.Counter()
            indices = set()
            for source in files:
                before = source.stat()
                blob = source.read_bytes()
                after = source.stat()
                assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
                row = json.loads(blob)
                ds = row['dataset']
                assert ds in expected
                assert prompt_ids[row['dataset_index']] == (ds, row['problem_id'])
                identity = (category, canonical, ds, row['problem_id'])
                assert identity not in all_seen, identity
                all_seen.add(identity)
                assert row['dataset_index'] not in indices
                indices.add(row['dataset_index'])
                relative = Path(category) / ds / canonical / source.name
                target = STAGE / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with target.open('xb') as f:
                        f.write(blob)
                digest = hashlib.sha256(blob).hexdigest()
                assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, str(target)
                assert target.stat().st_ino != source.stat().st_ino or target.stat().st_dev != source.stat().st_dev
                writer.writerow(dict(category=category, dataset=ds, config=canonical,
                                     source_config=source_config, k=config.get('k',''), threshold=config.get('threshold',''),
                                     problem_id=row['problem_id'], dataset_index=row['dataset_index'],
                                     finish_reason=row.get('finish_reason',''), response_tokens=len(row['response_token_ids']),
                                     relative_path=str(relative), source_path=str(source), sha256=digest, bytes=len(blob)))
                counts[ds] += 1
                finishes[row.get('finish_reason','unknown')] += 1
                manifest['total_rows'] += 1
                manifest['total_raw_bytes'] += len(blob)
            assert dict(counts) == expected, (source_config, counts)
            assert indices == set(prompt_ids)
            manifest['groups'].append(dict(category=category, config=canonical, source_config=source_config,
                                           counts=dict(counts), finish_reasons=dict(finishes)))
            index.flush()
            print(json.dumps(dict(category=category, config=canonical, rows=len(files), total=manifest['total_rows'])), flush=True)

assert manifest['total_rows'] == 118909
actual = sum(1 for category in SOURCES for _ in (STAGE/category).glob('*/*/*.json'))
assert actual == manifest['total_rows']
manifest['verification'] = dict(file_count=actual, all_copied_bytes_sha256_verified=True,
                                unique_problem_per_config=True, expected_dataset_counts_verified=True,
                                index_sha256=hashlib.sha256((STAGE/'index.csv').read_bytes()).hexdigest())
(STAGE / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
with (STAGE/'counts.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['category','dataset','config','rows'])
    for g in manifest['groups']:
        for ds,n in sorted(g['counts'].items()):w.writerow([g['category'],ds,g['config'],n])
(STAGE / 'README.md').write_text('''# 本次数学／推理与代码网格：单次 rollout 原始输出

目录结构：`<math或code>/<数据集>/<配置>/<原始dataset_index文件名>.json`。
每个 JSON 是一道题、一个配置的一次完整原始 rollout；原文件逐字节复制，保留 response、response_token_ids、prompt_token_ids、events、finish_reason 等原有字段。
这是独立普通文件副本，不依赖符号链接或硬链接。修改这里不会修改原实验文件。

## 范围

- math：20260912 网格，25 配置 × 2,254 题 = 56,350 条；含 MATH500、OlympiadBench、BBH。
- code：20260913 网格，25 配置加 student_only、teacher_only，共 27 配置 × 2,317 题 = 62,559 条；含 CRUXEval-I/O、HumanEval+、MBPP+、LiveCodeBench。
- 合计 118,909 条。生成超时的原始记录也保留，可用 finish_reason 区分。
- 本归档不含其他历史实验、AIME、DeepMath、dynamic-K；数学网格本身不含学生／教师基线。
- 不附加或重新计算答案评分。

配置名统一为 k1_t2、k1_t2.5 等。数学源目录中的 t2p5/t3p5 在本归档统一为 t2.5/t3.5，JSON 内容不变。
文件名保留原始 dataset_index，因此每个数据集目录内编号不一定从 00000 开始，也不必连续。

## 文件

- index.csv：逐条来源路径、归档相对路径、数据集、配置、problem_id、dataset_index、结束原因、token 数、SHA256。
- counts.csv：每个数据集、配置的记录数。
- manifest.json：来源、范围、数量、结束原因和校验结果。
- metadata/math 与 metadata/code：原实验配置、题目提示及已有数据清单等元数据副本。
- COMPLETE.json：全部复制和校验完成标记。

## 示例

读取 math/math500/k1_t2/ 下任意 JSON 的 response 字段，即可查看模型文本。
配置间按 dataset + problem_id 配对；不要仅依赖文件名跨实验关联。
''',encoding='utf-8')
(STAGE/'COMPLETE.json').write_text(json.dumps(dict(rows=actual, completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), manifest_sha256=hashlib.sha256((STAGE/'manifest.json').read_bytes()).hexdigest()),indent=2)+'\n')
os.rename(STAGE, DEST)
print(json.dumps(dict(destination=str(DEST), rows=actual, raw_bytes=manifest['total_raw_bytes'], verified=True)),flush=True)
