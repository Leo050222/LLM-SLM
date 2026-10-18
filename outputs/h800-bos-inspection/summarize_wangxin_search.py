import pathlib,json,collections,csv
ROOT=pathlib.Path(__file__).parent/'wangxin-full-search-20261011'
s=json.loads((ROOT/'summary.json').read_text());seen=set();groups=collections.defaultdict(lambda:{'objects':0,'nonempty':0,'bytes':0});user_roots=collections.defaultdict(lambda:{'objects':0,'nonempty':0,'bytes':0});eil=[];samples={};rows=[]
for f in sorted(ROOT.glob('*-matches.jsonl')):
 for line in f.open():
  o=json.loads(line);ident=(o['bucket'],o['key'])
  if ident in seen:continue
  seen.add(ident);rows.append(o);ps=o['key'].split('/');i=next(i for i,p in enumerate(ps) if 'wangxin' in p.casefold());prefix='/'.join(ps[:i+2]);g=o['bucket']+'/'+prefix;a=groups[g];a['objects']+=1;a['nonempty']+=o['size_bytes']>0;a['bytes']+=o['size_bytes']
  exact=next((i for i,p in enumerate(ps) if p.casefold()=='wangxin'),None)
  if exact is not None:
   name=o['bucket']+'/'+ '/'.join(ps[:exact+1])+'/';u=user_roots[name];u['objects']+=1;u['nonempty']+=o['size_bytes']>0;u['bytes']+=o['size_bytes']
  if 'eiloeo' in o['key'].casefold():eil.append(o)
  if '.cache/' not in o['key'] and len(samples.get(g,[]))<5:samples.setdefault(g,[]).append(o)
alias_path=ROOT/'westlake-alias-full-scan.json';alias=json.loads(alias_path.read_text()) if alias_path.exists() else None
alias_brief={k:alias[k] for k in ['bucket','prefix','status','objects_scanned','size_bytes','eiloeo_matches']} if alias else None
weight_samples=[]
if alias:
 for o in alias['targets']:
  k=o['key'].casefold()
  if k.endswith(('.safetensors','.ckpt')) and o['size_bytes']>1_000_000 and '/miniconda3/' not in k and '/venvs/' not in k:
   if len(weight_samples)<20:weight_samples.append(o)
r={'completed_buckets':s['completed_buckets'],'total_buckets':s['total_buckets'],'objects_scanned':s['objects_scanned'],'unique_matches':len(seen),'named_wangxin_directory_roots':dict(user_roots),'groups':dict(groups),'eiloeo_matches_in_wangxin_paths':eil,'westlake_related_alias':alias_brief,'alias_weight_filename_examples':weight_samples,'samples':samples,'read_only':True,'file_contents_scanned':False,'juicefs_full_directory_search':{'exit_code':0,'matches':0,'log':'../../h800-directory-inspection/codex-jfs-full-wangxin-20261011-145211-67cd/result.log'}}
(ROOT/'wangxin-results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
with (ROOT/'wangxin-matches.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['bucket','key','size_bytes','last_modified']);w.writeheader();w.writerows(rows)
for bucket in sorted({o['bucket'] for o in rows}):
 with (ROOT/(bucket+'-wangxin-matches.csv')).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['bucket','key','size_bytes','last_modified']);w.writeheader();w.writerows(o for o in rows if o['bucket']==bucket)
lines=['# wangxin 搜索结果','',f"本次检查当前凭据可见的 {s['total_buckets']} 个 BOS 桶，已完成 {s['completed_buckets']} 个。",f"按对象名称/路径检索 wangxin，不读取文件内容。累计检查 {s['objects_scanned']:,} 条对象元信息，去重后字符串命中 {len(seen):,} 个对象。",'大小按对象元信息相加，GB/TB 为十进制；不同路径下内容相同的副本没有按内容去重。','', '| 明确命名为 wangxin 的目录 | 对象数 | 总大小（GB） |','|---|---:|---:|']
for name,a in sorted(user_roots.items()):lines.append(f"| `bos://{name}` | {a['objects']:,} | {a['bytes']/1e9:.3f} |")
lines+=['','`auwomo-data-processed/wangxin/` 内有 `simforge/`、`libero_plus/`、`simforge-backups/` 和 `simforge-migration/`。另一个桶的 `auwomo-data-processed/wangxin/` 内有 simforge、备份及迁移包。','`bos://auwomo-data/wangxin_4090/` 只有零字节目录标记。','']
if alias:
 lines += [f"另发现相关旧实验目录 `bos://auwomo-data/wanxin_westlake/`（wanxin 少一个 g）：完整前缀检查已完成，共 {alias['objects_scanned']:,} 个对象，{alias['size_bytes']/1e12:.3f} TB。",'这里有 GenMol、LLaDA、MMaDA、REditor、RGroundBench、nli、模型检查点、运行环境和实验输出；其中的文件路径包含旧 `/storage/yukaichengLab/wangxin/GenMol/` 引用，以及 `pytest-of-wangxin` 临时目录。目录名和路径关联很强，但本次未核实所有文件的归属。','']
lines += ['GenMol `model_v2.ckpt` 和 LLaDA 一个模型分片的只读元信息查询成功；未下载权重内容或验证完整性。','已定位的 wangxin 命中路径及完整 wanxin_westlake 前缀中，未发现 eiloeo 名称；压缩包内部未检索。','当前 H800 PVC 的完整目录遍历（无深度限制）成功完成，wangxin 路径命中 0 个。BOS 中这些对象尚未在当前 /data 下找到对应目录。','', '完整对象清单按桶保存在 `auwomo-data-wangxin-matches.csv`、`auwomo-data-processed-wangxin-matches.csv`；逐桶扫描状态是 `summary.json`。','相关旧实验目录的完整名称检查记录是 `westlake-alias-full-scan.json`；权重元信息核对结果是 `westlake-alias-weight-head-check.json`。']
(ROOT/'wangxin-search-report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:r[k] for k in ['completed_buckets','objects_scanned','unique_matches','named_wangxin_directory_roots','westlake_related_alias']},ensure_ascii=False))
