#!/usr/bin/env python3
"""Rebuild a local-evidence experiment catalog without reading secrets or contacting servers.

Stdlib only. Run from any directory. Legacy source paths remain usable after the
project's reorganization because compatibility symlinks are retained.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import gzip
import io
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ['experiments', 'analysis', 'reports', 'outputs', 'figures', 'refine-logs', 'results/westlake']
SKIP_PARTS = {'node_modules', '__pycache__', '.git', '.mplcache', '.rollout_cache', 'cases', 'jobs', 'completed', 'scored', 'scores', 'raw', 'responses', 'requests', 'wangxin-full-search-20261011'}
SECRET = re.compile(r'password|passwd|api[_-]?key|access[_-]?key|secret|credential|authorization|private[_-]?key', re.I)
AGG_NAME = re.compile(r'summary|metrics|comparison|results|headline|aggregate|counts|analysis|pass8|training|verified|audit|complete|confidence|distribution|entropy|agreement|cohorts|quantiles|paired_changes|four_metrics|accuracy|horizon|correctness|per_dataset|paired_continuations|all_benchmarks|positions|frequency|robustness|coverage|cdf|sensitivity|bootstrap|relationship|recovery|support|buckets|regions|correlation|changes|length', re.I)
METRIC_KEY = re.compile(r'accuracy|^acc$|_acc$|correct|pass_at|pass_any|pass_all|all_8|all64|entropy|mean_gap|events|mean_tokens|mean_response_tokens|mean_output_tokens|rollouts_per_hour|tokens_per_second|reverse_kl|^kl$|rescue_rate|synergy_rate', re.I)
IDENTITY_KEYS = {'dataset', 'config', 'method', 'group', 'role', 'h', 'replicate', 'quartile', 'name', 'checkpoint_step', 'model_label', 'experiment', 'condition', 'mode', 'model', 'student', 'teacher'}
DATASETS = {'aime24', 'aime25', 'math500', 'olympiad_bench', 'bbh', 'cruxeval_i', 'cruxeval_o', 'humaneval_plus', 'mbpp_plus', 'livecodebench', 'deepmath'}

# These are associations to archived evidence, not a claim that raw remote files
# have been recovered. A shared report can legitimately cover several runs.
ASSOCIATIONS = {
 'baseline_math_gpt4o_mini_rescore_20260916':['reports/baseline_math_rescore_20260916'],
 'grid_math_gpt4o_mini_rescore_20260916':['reports/grid_math_rescore_comparison_20260916'],
 'code_grid_x8_20260916':['reports/code_grid_x8_results_20260918','reports/code_grid_x8_scoring_audit_20260918'],
 'code_grid_x8_score_20260918':['reports/code_grid_x8_results_20260918','reports/code_grid_x8_scoring_audit_20260918'],
 'code_grid_x8_score_parallel_20260918':['reports/code_grid_x8_results_20260918'],
 'code_reasoning_grid_20260913':['reports/extraction_audit_20260916'],
 'colocation_bench_20260918':['reports/colocation_bench_20260918'],
 'dynamic_k_entropy_20260914':['reports/dynamic_k_performance_20260915'],
 'full_teacher_student_x8_20260918':['reports/full_baseline_x8_status_20260918','reports/greedy_vs_temp1_20260921/temp1'],
 'full_teacher_student_x8_8gpu_20260918':['reports/full_baseline_x8_status_20260918','reports/greedy_vs_temp1_20260921/temp1'],
 'full_teacher_student_greedy_20260920':['reports/greedy_vs_temp1_20260921/greedy'],
 'full_teacher_student_greedy_8gpu_20260921':['reports/greedy_vs_temp1_20260921/greedy'],
 'full_teacher_student_greedy_score_20260921':['reports/greedy_vs_temp1_20260921/greedy'],
 'grid_async_20260916':['reports/grid_async_trial_20260916'],
 'grid_async_retry_20260916':['reports/grid_async_trial_20260916'],
 'grid_fifo_20260916':['reports/grid_fifo_fix_20260916'],
 'grid_math_4gpu_20260917':['reports/grid_math_resize_20260917'],
 'grid_math_8gpu_20260917':['reports/grid_math_resize_20260917'],
 'grid_math_colocated8_20260918':['reports/math_grid_x8_results_20260919'],
 'grid_math_colocated16_20260918':['reports/math_grid_x8_results_20260919'],
 'grid_math_colocated22_20260918':['reports/math_grid_x8_results_20260919'],
 'math_reasoning_grid_x8_20260916':['reports/math_grid_x8_results_20260919'],
 'math_grid_score_029041_20260919':['reports/math_grid_x8_results_20260919'],
 'math_grid_score_all_2000_20260919':['reports/math_grid_x8_results_20260919'],
 'intervention_analysis_greedy_20260921':['reports/intervention_analysis_greedy_20260921'],
 'intervention_analysis_phase1_20260921':['reports/intervention_analysis_phase1_20260921'],
 'logits_replay_20260918':['reports/logits_replay_20260918'],
 'logits_threshold2_replay_20260918':['reports/logits_threshold2_replay_20260918'],
 'main_greedy_k1_t2_20260921':['reports/main_greedy_comparison_20260921','reports/current_greedy_triplet_analysis_20260923'],
 'main_greedy_score_20260921':['reports/main_greedy_comparison_20260921'],
 'prefix_takeover_caps_20260917':['reports/prefix_takeover_caps_20260917','reports/prefix_caps_analysis_20260917'],
 'prefix_takeover_caps5_10_20260917':['reports/prefix_takeover_caps5_10_20260917'],
 'prefix_takeover_caps11_20_20260917':['reports/prefix_caps0_20_analysis_20260918'],
 'prefix_takeover_caps25_40_20260918':['reports/prefix_caps0_40_analysis_20260918'],
 'prefix_takeover_caps25_40_split_20260918':['reports/prefix_caps0_40_analysis_20260918'],
 'quartile_greedy_baselines_20260918':['reports/quartile_support_k_results_20260919'],
 'quartile_support_k_20260918':['reports/quartile_support_k_results_20260919'],
 'random_quartile_k1_20260918':['reports/random_quartile_comparison_20260918'],
 'reverse_t2_k1_20260918':['reports/reverse_t2_k1_baseline_comparison_20260918.json','reports/reverse_t2_k1_token_stats_20260918.json'],
 'reverse_t2_k1_score_20260918':['reports/reverse_t2_k1_baseline_comparison_20260918.json'],
 'reverse_greedy_t2_k1_8gpu_20260921':['reports/reverse_greedy_results_20260921'],
 'reverse_greedy_score_20260921':['reports/reverse_greedy_results_20260921'],
 'shift5_k1_t2_20260918':['reports/shift5_comparison_20260919'],
 'shift5_greedy_k1_t2_20260921':['reports/shift5_greedy_score_20260921'],
 'shift5_greedy_score_20260921':['reports/shift5_greedy_score_20260921'],
 'strong4_weak17_reprompt_greedy_t2_k1_20261005':['experiments/strong_reprompt_score_20261005/results','reports/strong_reprompt_vs_greedy_20261005','reports/strong_main_logit_analysis_20261006'],
 'strong8_weak17_reprompt_greedy_t2_k1_20261005':['experiments/strong_reprompt_score_20261005/results','reports/strong_reprompt_vs_greedy_20261005','reports/strong_main_logit_analysis_20261006'],
 'weak_reprompt_nodraft_t2_k1_20260920':['reports/weak_reprompt_nodraft_results_20260920'],
 'weak_reprompt_nodraft_score_20260920':['reports/weak_reprompt_nodraft_results_20260920'],
 'weak_reprompt_clean_t2_k1_20260921':['reports/weak_reprompt_clean_results_20260921'],
 'weak_reprompt_clean_score_20260921':['reports/weak_reprompt_clean_results_20260921'],
 'weak_reprompt_clean_greedy_t2_k1_20260921':['reports/weak_reprompt_clean_greedy_results_20260921'],
 'weak_reprompt_clean_greedy_score_20260921':['reports/weak_reprompt_clean_greedy_results_20260921'],
 'weak_reprompt_nodraft_greedy_t2_k1_20260921':['reports/weak_reprompt_nodraft_greedy_results_20260921'],
 'weak_reprompt_nodraft_greedy_score_20260921':['reports/weak_reprompt_nodraft_greedy_results_20260921'],
 'weak_resample_t2_k1_20260920':['reports/weak_resample_results_20260921'],
 'weak_resample_score_20260921':['reports/weak_resample_results_20260921'],
 'weak_resample_greedy_t2_k1_20260921':['reports/weak_resample_greedy_results_20260921'],
 'weak_resample_greedy_8gpu_20260921':['reports/weak_resample_greedy_results_20260921'],
 'weak_resample_greedy_score_20260921':['reports/weak_resample_greedy_results_20260921'],
 'qwen3_greedy_hard_three_20261005':['reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json'],
 'qwen3_greedy_remaining_20261005':['reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json'],
}

SUPPLEMENTARY = {
 'early_aime24_audit':('早期 AIME24 单次评测与提取审计',['EXPERIMENT_AUDIT.json','EXPERIMENT_AUDIT.md']),
 'deepmath_train25_top16_rkl':('DeepMath train25% / Top16 Reverse-KL 训练',['analysis/deepmath_train25_top16_rkl_logs']),
 'deepmath_heldout_checkpoints':('DeepMath thinking 测试 1000 题与五个 checkpoint',['results/legacy/deepmath-heldout-1000']),
 'deepmath_nonthinking_opd_comparison':('DeepMath nonthinking Student / Traditional / Bridge / Good-Bridge OPD',['figures/testset_model_comparison_20260908.json']),
 'deepmath_baseline_gap':('DeepMath thinking / nonthinking 逐 token teacher-student gap',['reports/deepmath_baseline_sample_gap/summary.json','reports/deepmath_nonthinking_sample_gap/summary.json','reports/deepmath_baseline_gap_position_analysis.json','reports/deepmath_student_top16_trace_summary.json']),
 'bridge_horizon_analysis':('Bridge horizon 与长度/准确率关系分析',['analysis/bridge_horizon_summary_20260904.csv','analysis/bridge_horizon_summary_20260906.csv','analysis/rollout_shortening_accuracy_20260907','analysis/teacher_student_length_accuracy_correlation_20260908.json','analysis/bridge_pass_metrics_20260906.csv']),
 'anchor_bifurcation_judging':('H256 分叉锚点语义判定（Luna）',['outputs/anchor_bifurcation_h256_openlux_luna_seed20260908/full/run_summary.json','outputs/anchor_bifurcation_h256_openlux_luna_seed20260908/pilot8/run_summary.json']),
 'main_grid_x8_plot_data':('八轮主结果绘图数据与校验',['figures/main_grid_x8_20260920']),
 'intervention_position_analysis':('介入位置、长度、分布与三元结果分析',['reports/intervention_positions_20260923','reports/k1_t2_positions_20260916','reports/k1_t2_length_vs_student_20260916','reports/intervention_distribution_20260911','reports/current_greedy_triplet_analysis_20260923']),
 'all_benchmark_comparison':('早期单次 rollout 跨 benchmark 比较',['reports/all_benchmark_comparison_20260911']),
 'benchmark_visualization':('全实验可视化数据与混合策略三元表',['outputs/benchmark-viz','outputs/triple-outcomes','outputs/prefix-triple-outcomes','outputs/01a085ee-f483-7b63-83aa-0ac0dbf0fac4/merged_benchmarks']),
 'mechanism_counterfactual_plan':('B3/B4 同前缀反事实与机制收益归属（计划）',['refine-logs/EXPERIMENT_PLAN.md','refine-logs/EXPERIMENT_TRACKER.md']),
 'early_method_plans':('早期方法与训练、介入长度、锚点确认实验规划',['refine-logs','idea-stage']),
}


@lru_cache(maxsize=None)
def rel(p): return p.relative_to(ROOT).as_posix()
def within(path, prefix): return path == prefix or path.startswith(prefix.rstrip('/') + '/')
def safe_string(v):
 s=str(v)
 if len(s)>160 or SECRET.search(s) or '://' in s or '\n' in s: return ''
 if '/' in s: return Path(s).name
 return s

def eligible(p):
 r=p.relative_to(ROOT)
 if any(x in SKIP_PARTS for x in r.parts) or any(x.startswith('h800-') or x.startswith('h800_') for x in r.parts): return False
 if p.suffix.lower() not in {'.json','.csv','.md'}:return False
 if p.name in {'prompts.jsonl','per_question.json'} or SECRET.search(p.name):return False
 if p.stat().st_size>12_000_000:return False
 if p.suffix=='.md': return len(r.parts)<5
 return bool(AGG_NAME.search(p.stem)) or within(r.as_posix(),'results/legacy/deepmath-heldout-1000') or p.name=='EXPERIMENT_AUDIT.json'

def flatten_numeric(obj, pointer='', context=None):
 context=dict(context or {})
 if isinstance(obj, dict):
  for k in sorted(IDENTITY_KEYS):
   if k in obj and isinstance(obj[k],(str,int,float)) and not isinstance(obj[k],bool):
    v=safe_string(obj[k])
    if v:context[k]=v
  for k,v in obj.items():
   if SECRET.search(k):continue
   token=k.replace('~','~0').replace('/','~1')
   c=dict(context)
   if k in DATASETS: c['dataset']=k
   elif re.fullmatch(r'k\d+_t\d+(?:p\d+)?',k):c['config']=k
   elif k in {'student','teacher','online','prefix256','student64','h32','h64','h128','h256'}:c['method']=k
   if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v):
    yield {'pointer':pointer+'/'+token,'metric':k,'value':v,'context':c}
   elif isinstance(v,(dict,list)):yield from flatten_numeric(v,pointer+'/'+token,c)
 elif isinstance(obj,list):
  # Arrays of scalar token IDs / raw observations are not aggregate metrics.
  for i,v in enumerate(obj):
   if isinstance(v,(dict,list)):yield from flatten_numeric(v,pointer+'/'+str(i),context)

def metric_records(obj,pointer='',context=None):
 context=dict(context or {})
 if isinstance(obj,dict):
  for k in sorted(IDENTITY_KEYS):
   if k in obj and isinstance(obj[k],(str,int,float)) and not isinstance(obj[k],bool):
    value=safe_string(obj[k])
    if value:context[k]=value
  nums={k:v for k,v in obj.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and not SECRET.search(k)}
  if any(METRIC_KEY.search(k) for k in nums):yield {'pointer':pointer or '/','context':context,'metrics':nums}
  for k,v in obj.items():
   if SECRET.search(k):continue
   c=dict(context)
   if k in DATASETS:c['dataset']=k
   elif re.fullmatch(r'k\d+_t\d+(?:p\d+)?',k):c['config']=k
   elif k in {'student','teacher','online','prefix256','student64','h32','h64','h128','h256'}:c['method']=k
   yield from metric_records(v,pointer+'/'+k.replace('~','~0').replace('/','~1'),c)
 elif isinstance(obj,list):
  for i,v in enumerate(obj):
   if isinstance(v,(dict,list)):yield from metric_records(v,pointer+'/'+str(i),context)

def load_json(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None

def fmt(v):
 if isinstance(v,float):return f'{v:.6g}'
 return str(v)

def source_link(path):
 p=ROOT/path
 if p.is_file() and p.stat().st_size>5*1024*1024:
  return f'`{path}`（大文件保留在工作区；版本指纹见机器清单）'
 return f'[{path}](../../{path})'
def number_cell(record):
 c=record['context'];m=record['metrics']
 ident=' / '.join(f'{k}={v}' for k,v in c.items() if k in {'dataset','config','method','role','h','quartile','replicate','group','checkpoint_step','experiment','condition','model_label'})
 chosen=[]
 for k in ['n','questions','rows','rollouts','correct','correct_rollouts','accuracy','accuracy_pct','acc','pass_at_8','pass_at_8_pct','pass_at_64','pass_any','mean_output_tokens','mean_response_tokens','mean_tokens','mean_interventions','teacher_token_fraction','teacher_token_pct','events','mean_gap']:
  if k in m: chosen.append(f'{k}={fmt(m[k])}')
 if not chosen: chosen=[f'{k}={fmt(v)}' for k,v in list(m.items())[:4]]
 return (ident+'；' if ident else '')+'，'.join(chosen[:7])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='docs/experiments');args=ap.parse_args()
 out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
 all_files=[]
 for base in SCOPE+['results/legacy/deepmath-heldout-1000','idea-stage']:
  d=ROOT/base
  if d.exists():all_files.extend(p for p in d.rglob('*') if p.is_file())
 for f in ['EXPERIMENT_AUDIT.json','EXPERIMENT_AUDIT.md']:
  if (ROOT/f).exists():all_files.append(ROOT/f)
 files=sorted({rel(p):p for p in all_files}.values(),key=rel)
 selected=[p for p in files if eligible(p)]
 artifacts=[];scalars=[];records=[];parse_errors=[]
 for p in selected:
  r=rel(p);newrecords=[];newscalars=[]
  if p.suffix=='.json':
   data=load_json(p)
   if data is None:parse_errors.append(r)
   else:
    newrecords=list(metric_records(data));newscalars=list(flatten_numeric(data))
  elif p.suffix=='.csv':
   try:
    with p.open(newline='') as f:
     rows=list(csv.DictReader(f))
    # Long per-sample tables stay referenced as artifacts; they are not
    # re-labelled as aggregate results or duplicated into the catalog.
    if len(rows)<=2500:
     for i,row in enumerate(rows):
      obj={}
      for k,v in row.items():
       if k is None or SECRET.search(k):continue
       try:
        n=float(v)
        if math.isfinite(n):obj[k]=int(n) if n.is_integer() else n
       except (ValueError,TypeError):
        if k in IDENTITY_KEYS:obj[k]=safe_string(v)
      newrecords.extend(metric_records(obj,'/rows/'+str(i)))
      newscalars.extend(flatten_numeric(obj,'/rows/'+str(i)))
   except (OSError,csv.Error,UnicodeDecodeError):parse_errors.append(r)
  for x in newrecords:x['source']=r
  for x in newscalars:x['source']=r
  records+=newrecords;scalars+=newscalars
  artifacts.append({'path':r,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'kind':p.suffix[1:],'aggregate_records':len(newrecords),'numeric_values':len(newscalars)})
 by_source=defaultdict(list)
 for x in records:by_source[x['source']].append(x)
 expdirs=sorted(d for d in (ROOT/'experiments').iterdir() if d.is_dir())
 series=[]
 for d in expdirs:
  name=d.name;path=rel(d)
  own=[a['path'] for a in artifacts if within(a['path'],path)]
  remote_roots=sorted(rel(p) for p in (ROOT/'results/westlake').glob('*/*') if p.is_dir() and p.name==name) if (ROOT/'results/westlake').exists() else []
  associated=ASSOCIATIONS.get(name,[])+remote_roots
  related=[a['path'] for a in artifacts if any(within(a['path'],x) for x in associated)]
  result_sources=sorted({x for x in own+related if by_source[x]})
  # Marker files establish only the stage named by the marker.
  markers=sorted(p for p in own+related if 'COMPLETE' in Path(p).name or 'VERIFIED' in Path(p).name)
  local_fs=[p for p in files if within(rel(p),path)]
  config_sources=sorted(rel(p) for p in local_fs if p.name in {'configs.json','config.json','PLAN.json','README.md','STATUS.md','PLAN.md','LAUNCH.md','submission.json','LAUNCH.json'})
  if result_sources:status='result_summary_available'
  elif any(isinstance(x,dict) and (str(x.get('status','')).lower() in {'blocked','failed'} or x.get('blocked') is True) for x in (load_json(p) for p in local_fs if p.name=='SUBMISSION_BLOCKED.json')):status='blocked_or_failed_submission'
  elif any(p.name in {'LAUNCH.json','submission.json','SUBMITTED.json','LAUNCH.md'} for p in local_fs):status='deployment_record_only'
  elif any(p.name in {'PLAN.md','PLAN.json'} for p in local_fs):status='planned_or_code_only'
  else:status='code_or_status_only'
  marker_objects=[load_json(ROOT/m) for m in markers]
  if any(isinstance(m,dict) and m.get('archive_verified') is True and m.get('events',0)>0 and m.get('semantic_annotation_completed') is False for m in marker_objects):status='analysis_complete_semantic_and_counterfactual_partial'
  if any(isinstance(m,dict) and m.get('checksum_all_files') is True and m.get('dtype')=='BF16' and m.get('roles') for m in marker_objects):status='archive_verified_payload_kept_on_westlake'
  series.append({'id':name,'path':path,'status':status,'local_file_count':len(local_fs),'local_json_count':sum(p.suffix=='.json' for p in local_fs),'config_sources':config_sources,'result_sources':result_sources,'completion_markers':markers,'associated_evidence_roots':associated,'aggregate_records':sum(len(by_source[s]) for s in result_sources),'notes':'汇总表/阶段完成标记不证明远端原始输出已迁回；同一共享报告可能覆盖多个运行。'})
 for sid,(title,roots) in SUPPLEMENTARY.items():
  sources=sorted(a['path'] for a in artifacts if any(within(a['path'],r) for r in roots))
  result_sources=[s for s in sources if by_source[s]]
  status='planned_only' if sid in {'mechanism_counterfactual_plan','early_method_plans'} else ('result_summary_available' if result_sources else 'report_only')
  series.append({'id':sid,'title':title,'path':roots[0],'status':status,'local_file_count':sum(any(within(rel(p),r) for r in roots) for p in files),'local_json_count':sum(p.suffix=='.json' and any(within(rel(p),r) for r in roots) for p in files),'config_sources':roots,'result_sources':result_sources,'completion_markers':[],'associated_evidence_roots':roots,'aggregate_records':sum(len(by_source[s]) for s in result_sources),'notes':'补充系列可与顶层实验或其他分析共享数据；不应加总为独立样本。'})
 remote_series=[]
 remote_root=ROOT/'results/westlake'
 if remote_root.exists():
  for d in sorted(remote_root.glob('*/*')):
   if not d.is_dir():continue
   prefix=rel(d)
   sources=sorted(a['path'] for a in artifacts if within(a['path'],prefix))
   result_sources=[x for x in sources if by_source[x]]
   markers=[x for x in sources if 'COMPLETE' in Path(x).name or 'VERIFIED' in Path(x).name]
   status='result_summary_available' if result_sources else 'metadata_only'
   ownfiles=[p for p in files if within(rel(p),prefix)]
   if not result_sources:
    explicit_states=[load_json(p) for p in ownfiles if p.suffix=='.json' and ('FAILED' in p.name or 'BLOCKED' in p.name)]
    if any(isinstance(x,dict) and str(x.get('status','')).lower() in {'failed','blocked','error'} for x in explicit_states):status='failed_attempt'
   remote_series.append({'id':'westlake/'+d.parent.name+'/'+d.name,'title':'西湖 '+d.parent.name+'/'+d.name,'path':prefix,'status':status,'local_file_count':len(ownfiles),'local_json_count':sum(p.suffix=='.json' for p in ownfiles),'config_sources':[prefix],'result_sources':result_sources,'completion_markers':markers,'associated_evidence_roots':[prefix],'aggregate_records':sum(len(by_source[x]) for x in result_sources),'notes':'远端小摘要同步；可能与本地系列为同一运行。大型原始输出保留西湖。'})
 series+=remote_series
 inventory=load_json(ROOT/'artifacts/storage_inventory/westlake_inventory_20261019.json') or {}
 status_counts=Counter(s['status'] for s in series[:len(expdirs)])
 coverage={'snapshot_date':'2026-10-19 Asia/Shanghai','scope':SCOPE+['results/legacy/deepmath-heldout-1000','idea-stage','EXPERIMENT_AUDIT.json','EXPERIMENT_AUDIT.md'],'top_level_experiment_directories':len(expdirs),'cataloged_top_level_experiment_directories':len(series[:len(expdirs)]),'supplementary_series':len(SUPPLEMENTARY),'westlake_run_directories':len(remote_series),'westlake_synced_files':sum(within(rel(p),'results/westlake') for p in files),'westlake_top_level_inventory_entries':len(inventory.get('top_level',[])),'files_seen':len(files),'indexed_artifacts':len(artifacts),'aggregate_records':len(records),'numeric_values':len(scalars),'status_counts':dict(status_counts),'parse_errors':parse_errors,'remote_recovery':'2026-10-19 已连接 gpu-wangxin 西湖旧共享存储 /storage/yukaichengLab/wangxin/test_eiloeo；小结果汇总和 manifest 下载到 results/westlake。大型 checkpoints/rollouts/logits 保留在西湖；新 PVC 的迁移缺口不代表西湖原盘缺失。'}
 for item in series:
  candidates=[]
  preferred=sorted(item['result_sources'],key=lambda x:(0 if Path(x).name in {'summary.json','comparison.json','full_results_summary.json'} else 1,len(x),x))
  for source in preferred:
   candidates.extend(r for r in by_source[source] if any(re.fullmatch(r'accuracy|accuracy_pct|acc|pass_at_8|pass_at_64|mean_gap|events|tokens_per_second|rollouts_per_hour',k) for k in r['metrics']))
   if len(candidates)>=8:break
  item['key_results']=candidates[:8]
 payload={'schema_version':1,'coverage':coverage,'series':series,'artifacts':artifacts,'records_file':'result_records.csv.gz','records_schema':['source','pointer','context_json','metrics_json'],'numeric_metrics_file':'numeric_metrics.csv.gz'}
 (out/'experiment_catalog.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 def write_csv(name,fields,rows):
  if name.endswith('.gz'):
   with (out/name).open('wb') as raw:
    with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as compressed:
     with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
  else:
   with (out/name).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 write_csv('experiment_series.csv',['id','title','path','status','local_file_count','local_json_count','aggregate_records','config_sources','result_sources','completion_markers','notes'],[{k:json.dumps(s.get(k),ensure_ascii=False) if isinstance(s.get(k),list) else s.get(k,'') for k in ['id','title','path','status','local_file_count','local_json_count','aggregate_records','config_sources','result_sources','completion_markers','notes']} for s in series])
 write_csv('result_records.csv.gz',['source','pointer','context_json','metrics_json'],[{'source':r['source'],'pointer':r['pointer'],'context_json':json.dumps(r['context'],ensure_ascii=False,sort_keys=True),'metrics_json':json.dumps(r['metrics'],ensure_ascii=False,sort_keys=True)} for r in records])
 write_csv('numeric_metrics.csv.gz',['source','pointer','context_json','metric','value'],[{'source':r['source'],'pointer':r['pointer'],'context_json':json.dumps(r['context'],ensure_ascii=False,sort_keys=True),'metric':r['metric'],'value':r['value']} for r in scalars])
 key_results=[{'source':r['source'],'pointer':r['pointer'],'context_json':json.dumps(r['context'],ensure_ascii=False,sort_keys=True),'metrics_json':json.dumps(r['metrics'],ensure_ascii=False,sort_keys=True)} for item in series for r in item['key_results']]
 write_csv('key_results.csv',['source','pointer','context_json','metrics_json'],key_results)
 write_csv('result_artifacts.csv',['path','bytes','sha256','kind','aggregate_records','numeric_values'],artifacts)
 (out/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
 write_report(out,series,coverage,by_source)
 print(json.dumps(coverage,ensure_ascii=False,indent=2))

STATUS_ZH={'result_summary_available':'已有结果汇总','blocked_or_failed_submission':'提交阻塞/失败记录','deployment_record_only':'仅定位部署记录','planned_or_code_only':'仅计划/代码证据','code_or_status_only':'仅代码/状态文档','analysis_complete_semantic_and_counterfactual_partial':'数值分析完成；语义/反事实未完','archive_verified_payload_kept_on_westlake':'归档校验完成；大载荷保留西湖','planned_only':'仅计划','report_only':'仅文档证据','metadata_only':'仅元数据/阶段标记','failed_attempt':'失败尝试记录'}


def write_report(out,series,cov,by_source):
 lines=['# 全项目实验与结果总目录','',
 '证据快照：2026-10-19（Asia/Shanghai）。本文件由 `tools/organization/build_experiment_catalog.py` 从本地保存的 JSON、CSV 和阶段文档生成。','',
 f'覆盖 `experiments/` 下 **{cov["top_level_experiment_directories"]}/{cov["top_level_experiment_directories"]} 个顶层目录**，另有 **{cov["supplementary_series"]} 个早期/跨目录系列**与 **{cov['westlake_run_directories']} 个西湖运行目录**；扫描 {cov["files_seen"]:,} 个文件，索引 {cov["indexed_artifacts"]:,} 个汇总或文档产物，保留 {cov["aggregate_records"]:,} 条聚合记录与 {cov["numeric_values"]:,} 个原始数值。','',
 '**存储范围：** 2026-10-19 已连接西湖旧共享存储 `gpu-wangxin:/storage/yukaichengLab/wangxin/test_eiloeo`。本次同步的小结果汇总和 manifest 位于 `results/westlake/`；大型 checkpoints、全量 rollouts 与 BF16 logits 继续保留在西湖。新 H800 PVC 的迁移状态与西湖原盘是两件事，不据此认定原始数据缺失。目录列出本地文件与同步摘要能够核实的实验，不能仅凭摘要声明每个远端原始文件都完成复验。','',
 '**口径：** `accuracy` 通常为 0–1，`accuracy_pct` 为 0–100；保留源字段名与单位。八轮采样的 seed/replicate 标准差不等于多次独立训练标准差。`pass@8`、`all@8` 与 rollout 平均准确率分别列示。共享报告、迁移/扩容目录和重评分目录可能指向同一批样本，不能相加为独立实验数。超时的最终答案正确与严格完成率也不能混用。','',
 '## 文件入口','',
 '- [experiment_series.csv](experiment_series.csv)：逐系列状态、配置入口、结果源和完成标记。',
 '- [experiment_catalog.json](experiment_catalog.json)：机器清单、源 SHA256 与每系列关键数字；[key_results.csv](key_results.csv) 可直接打开。',
 '- [result_records.csv.gz](result_records.csv.gz)：完整聚合结果表（gzip 压缩），保留原字段；`pointer` 定位 JSON 对象或 CSV 数据行（从 0 起）。',
 '- [numeric_metrics.csv.gz](numeric_metrics.csv.gz)：汇总产物内数值长表（gzip 压缩），不复制原始回答或凭据。',
 '- [result_artifacts.csv](result_artifacts.csv)、[coverage.json](coverage.json)：产物摘要与覆盖统计。','',
 '## 可以直接复核的主要结果','',
 '下面只对明确匹配的配置比较数值，不根据计划推断结果。','']
 # DeepMath held-out checkpoint results.
 lines+=['### 早期 DeepMath thinking held-out 1000 与 checkpoint','',
 '| 条件 | 正确/题数 | accuracy | 相对 baseline 变化（百分点） | 平均输出 tokens | 证据 |',
 '|---|---:|---:|---:|---:|---|']
 westlake_baseline='results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/merged/summary.json'
 base=load_json(ROOT/westlake_baseline) or load_json(ROOT/'results/legacy/deepmath-heldout-1000/baseline.json') or {}
 for name in ['baseline','step27','step53','step80','step106','step133']:
  source=westlake_baseline if name=='baseline' else f'results/westlake/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_{name}_seed20260811/merged/summary.json'
  if not (ROOT/source).exists():source=f'results/legacy/deepmath-heldout-1000/{name}.json'
  d=load_json(ROOT/source)
  if not d:continue
  acc=d.get('accuracy');correct=d.get('correct',d.get('checkpoint_correct'));n=d.get('rows',1000)
  lines.append(f'| {name} | {correct}/{n} | {acc:.4%} | {(acc-base.get("accuracy",acc))*100:+.2f} | {d.get("mean_response_tokens",0):.2f} | {source_link(source)} |')
 lines+=['','这是 thinking 协议；下表是 nonthinking 协议，两个表的准确率不可直接视为同协议训练收益。DeepMath 的历史评分后来进行了布尔答案归一化，下面 OPD 四模型表保留 2026-09-08 保存版本；不要与 2026-09-11 的修订分数交叉比较。','',
 '### Nonthinking OPD 四种模型（相同测试 1000 题，每题 8 次）','',
 '| 模型 | rollout 数 | 正确数 | 平均准确率 | pass@8 | all@8 | 相对 Student（百分点 / 相对比例） |','|---|---:|---:|---:|---:|---:|---:|']
 source='figures/testset_model_comparison_20260908.json';d=load_json(ROOT/source) or {};methods=d.get('methods',[]);baseacc=methods[0]['accuracy'] if methods else 1
 for x in methods:
  a=x['accuracy'];lines.append(f'| {x["name"]}（step {x["checkpoint_step"]}） | {x["rollouts"]} | {x["correct_rollouts"]} | {a:.4%} | {x["pass_at_8"]:.2%} | {x["all_8_correct_rate"]:.2%} | {(a-baseacc)*100:+.3f} / {(a/baseacc-1)*100:+.2f}% |')
 lines+=['',f'数据源：{source_link(source)}。这里报告保存的单模型 checkpoint 评测；未把评测重复当作独立训练种子。','',
 '### 主方法 dual-greedy k=1、threshold=2（八个 benchmark）','',
 '| 数据集 | n | Student | Hybrid | Teacher | Hybrid − Student（百分点） | 相对提升 |','|---|---:|---:|---:|---:|---:|---:|']
 source='reports/main_greedy_comparison_20260921/four_metrics.json';d=load_json(ROOT/source) or [];groups=defaultdict(dict)
 for x in d:groups[x['dataset']][x['method']]=x
 for dataset in ['math500','olympiad_bench','bbh','cruxeval_i','cruxeval_o','humaneval_plus','mbpp_plus','livecodebench','Overall']:
  g=groups.get(dataset,{})
  s=g.get('student');t=g.get('teacher');h=g.get('k1_t2') or g.get('hybrid') or g.get('main')
  if not h:
   possible=[x for k,x in g.items() if k not in ['student','teacher']]
   h=possible[0] if len(possible)==1 else None
  if s and h and t:
   sa=s['accuracy_pct'];ha=h['accuracy_pct'];ta=t['accuracy_pct']
   lines.append(f'| {dataset} | {h["n"]} | {sa:.3f}% | {ha:.3f}% | {ta:.3f}% | {ha-sa:+.3f} | {(ha/sa-1)*100:+.2f}% |')
 lines +=['',f'数据源：{source_link(source)}；严格 scoring 与 timeout 规则沿用源汇总。Teacher 强于 Student 的配对不代表每题均强于 Student。','',
 '### 八轮 grid 与消融结果的入口','',
 '| 系列 | 已保存结果与范围 | 源文件 |','|---|---|---|',
 '| 数学/BBH 25 组 grid × 8 | MATH500、OlympiadBench、BBH；含 seed/replicate SD、pass@8、all@8、超时和 teacher token 占比 | '+source_link('reports/math_grid_x8_results_20260919/summary.json')+' |',
 '| 代码五 benchmark grid × 8 | CRUXEval-I/O、HumanEval+、MBPP+、LiveCodeBench；汇总评分 463,400 行（25 组 × 五数据集 × 八轮；baseline 另表） | '+source_link('reports/code_grid_x8_results_20260918/x8_score_summary.json')+' |',
 '| prefix 接管预算 0–20 / 25–40 | 每数据集固定 sample100 × 8；与 teacher-only 配对比较 | '+source_link('reports/prefix_caps0_20_analysis_20260918/summary.json')+'；'+source_link('reports/prefix_caps0_40_analysis_20260918/metrics.json')+' |',
 '| 随机时机、shift5、reverse、weak reprompt/resample | 分开保留 greedy 与 temp1；详细准确率见逐系列结果 | '+source_link('reports/random_quartile_comparison_20260918/summary.json')+' |',
 '| Qwen3 1.7B / 4B / 8B / 14B | 八 benchmark matched greedy；AIME 中 1.7B/14B 使用历史 T=1，需单列协议 | '+source_link('reports/qwen3_1p7b_4b_8b_14b_all_results_20261005.json')+' |','',
 '### 介入机制分析的已完成部分与边界','',
 '| 协议 | 事件数 | 涉及题数 | teacher token 落在 student top5 | 高支持 / 低支持 | 平均弱熵 − 强熵 |','|---|---:|---:|---:|---:|---:|']
 for label,path in [('temp1 × 8','reports/intervention_analysis_phase1_20260921/headline.json'),('greedy','reports/intervention_analysis_greedy_20260921/headline.json')]:
  d=load_json(ROOT/path) or {};x=d.get('pooled_event',{})
  if x:lines.append(f'| {label} | {x["events"]:,} | {x["questions"]:,} | {x["top5"]:.3%} | {x["high"]:.3%} / {x["low"]:.3%} | {x["delta"]:.6f} nats |')
 lines+=['','源文件：'+source_link('reports/intervention_analysis_phase1_20260921/headline.json')+'、'+source_link('reports/intervention_analysis_greedy_20260921/headline.json')+'。这些是描述统计；120 例语义盲标复核尚未完成，B3/B4 同前缀反事实生成仍是计划。不能据此声称已证明“选中了正确逻辑分支”或完成因果机制识别。','',
 '## 全部顶层实验目录','',
 '每个目录列一个索引，包含部署、评分、扩容和复用阶段。状态“已有结果汇总”指至少找到数值证据；不保证该系列全部配置都完成。具体完成范围应以对应 `ALL_*_COMPLETE` / audit 标记和结果行数复核。没有找到汇总时，保持未确认，不用 README 的运行计划补齐结果。','',
 '| 编号 | 实验目录 | 证据状态 | 本地文件数 | 聚合记录数 |','|---:|---|---|---:|---:|']
 n=cov['top_level_experiment_directories']
 for i,s in enumerate(series[:n],1):lines.append(f'| {i} | [{s["id"]}](#{s["id"].lower()}) | {STATUS_ZH[s["status"]]} | {s["local_file_count"]:,} | {s["aggregate_records"]:,} |')
 for i,s in enumerate(series[:n],1):
  lines+=['',f'### {s["id"]}','',f'- 证据状态：**{STATUS_ZH[s["status"]]}**；代码目录：{source_link(s["path"])}。']
  if s['config_sources']:lines.append('- 配置/范围入口：'+'、'.join(source_link(p) for p in s['config_sources'][:5])+'。')
  if s['completion_markers']:lines.append('- 已保存阶段完成标记：'+'、'.join(source_link(p) for p in s['completion_markers'][:5])+'。')
  if s['result_sources']:
   # Prefer final summary over repeated CSV mirrors, then show representative
   # per-dataset metrics with their exact field names.
   ranked=sorted(s['result_sources'],key=lambda p:(0 if Path(p).name in ['summary.json','summary_with_baselines.json','full_results_summary.json','results.json','comparison.json','combined_completed_metrics.json'] else 1,0 if p.endswith('.json') else 1,len(p),p))
   lines.append('- 结果入口：'+'、'.join(source_link(p) for p in ranked[:5])+'。')
   preview=[]
   for src in ranked:
    useful=[r for r in by_source[src] if any(re.fullmatch(r'accuracy|accuracy_pct|acc|pass_at_8|pass_at_64|mean_gap|events|tokens_per_second|rollouts_per_hour',k) for k in r['metrics'])]
    if useful:
     preview=useful;break
   if preview:
    lines+=['','| 已保存聚合记录（节选；字段与单位沿用源文件） | 来源位置 |','|---|---|']
    for r in preview[:8]:lines.append('| '+number_cell(r).replace('|','\\|')+' | '+source_link(r['source'])+' `'+r['pointer']+'` |')
    if len(preview)>8:lines.append(f'\n本源另有 {len(preview)-8} 条聚合记录；完整数值见 `result_records.csv.gz`。')
  else:lines.append('- 未定位最终数值汇总；当前仅保留配置、代码或历史状态。不能据此断言实验成功或失败。')
  if s['id'].startswith('intervention_logits_bf16_'):lines.append('- BF16 完成标记记录远端归档已校验；全部张量仍保留西湖，本地只同步小摘要和校验标记。')
  if s['id'].startswith('weak1p7_') and ('postlogit' in s['id'] or 'mainminus' in s['id']):lines.append('- 不把命名中的 mainminus/postlogit 当成已有结果；应另行核对完成输出。')
 lines+=['','## 不在 experiments/ 的早期与跨目录系列','']
 for s in series[n:n+cov['supplementary_series']]:
  lines += [f'### {s["title"]}','',f'状态：**{STATUS_ZH[s["status"]]}**。入口：'+'、'.join(source_link(p) for p in s['config_sources'])+'。']
  if s['result_sources']:
   lines.append('\n保存数值的产物：'+'、'.join(source_link(p) for p in s['result_sources'][:8])+'。')
  lines.append('')
 lines+=['## 西湖共享存储运行目录覆盖','',f'已同步 {cov["westlake_synced_files"]:,} 个小文件，覆盖 {cov["westlake_run_directories"]} 个运行/数据目录；远端顶层清点 {cov["westlake_top_level_inventory_entries"]} 项。这里只对已下载的小摘要建立索引，不复制数据集、模型权重和大型输出。远端清点见 '+source_link('artifacts/storage_inventory/westlake_inventory_20261019.json')+'。','', '| 远端分类/运行 | 证据状态 | 同步文件数 | 数值记录数 | 结果节选（源单位） | 汇总入口 |','|---|---|---:|---:|---|---|']
 for s in series[n+cov['supplementary_series']:]:
  src=s['result_sources'][:2] or s['config_sources']
  sample=[]
  for source in src:
   sample.extend(r for r in by_source[source] if any(re.fullmatch(r'accuracy|accuracy_pct|acc|pass_at_8|pass_at_64|mean_gap|events|tokens_per_second|rollouts_per_hour',k) for k in r['metrics']))
  excerpt='<br>'.join(number_cell(r).replace('|','\\|') for r in sample[:2]) or '未定位准确率/主指标汇总'
  lines.append('| '+s['id'].replace('westlake/','')+' | '+STATUS_ZH[s['status']]+' | '+str(s['local_file_count'])+' | '+str(s['aggregate_records'])+' | '+excerpt+' | '+'、'.join(source_link(p) for p in src)+' |')
 lines+=['','历史 DeepMath 评分存在布尔答案归一化修订；源汇总与 2026-09-11 统一比较表可能分别保存旧、新口径。本目录保留历史值和版本路径，不覆盖旧数值；跨协议/评分版本比较须回到原报告的 scoring 定义。较新的统一比较见 '+source_link('reports/all_benchmark_comparison_20260911/REPORT.md')+'。','']
 lines+=['## 结果解读与后续工作','',
 '1. Nonthinking held-out 1000 中，Good-Bridge OPD 相对 Student 的 rollout 准确率为 30.225% 对 28.0875%（+2.1375 个百分点，相对 +7.61%），但未保存多训练种子证据；不应仅凭八次采样解释训练稳定性。',
 '2. 主方法的 greedy 表已覆盖八个 benchmark；随机介入、延迟 shift5、reverse 与 reprompt/resample 均有独立结果源，不能仅保留有利子组。',
 '3. 同前缀熵和支持度结果只支持统计描述。要测试“路径修复”的解释，下一步是按已冻结计划补完语义盲标、反事实 pilot 和确认集；本次目录整理未启动新实验。',
 '4. 西湖已可连接；后续按需读取原始 rollouts、评分逐题文件、训练 checkpoints 和 logits payload，再核对 source SHA256、manifest 行数和完成标记。大型原始输出不随代码推送 GitHub。','',
 '## 再生成与校验','',
 '```bash','python3 tools/organization/build_experiment_catalog.py','```','',
 '脚本只读取本地证据并覆写本目录的生成文件；不访问网络、不发起评分、不提交训练。源路径使用整理前兼容路径，目录规整后由项目保留的 symlink 解析。机器清单保留每个索引源文件的 SHA256 便于核对版本。',
 f'本轮解析失败文件数：{len(cov["parse_errors"])}。' + (' 失败入口：' + '、'.join(source_link(p) for p in cov['parse_errors']) + '（空文件或无效 JSON，未补造内容）。' if cov['parse_errors'] else ''), '']
 (out/'ALL_EXPERIMENTS.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
