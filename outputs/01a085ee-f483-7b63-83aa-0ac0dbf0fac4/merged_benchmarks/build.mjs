import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const out=path.dirname(fileURLToPath(import.meta.url));
const d=JSON.parse(await fs.readFile(path.join(out,'data.json'),'utf8'));
const wb=Workbook.create();
const main=wb.worksheets.add('全部实验总表');
const detail=wb.worksheets.add('可筛选明细');
const notes=wb.worksheets.add('口径与来源');
const col=n=>{let s='';for(n++;n>0;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const label=g=>g==='student_only'?'Student-only':g==='teacher_only'?'Teacher-only':'无Luna';
const params=g=>g.endsWith('_only')?[null,null]:[Number(g.split('_t')[1]),Number(g.split('_')[0].slice(1))];
const map=new Map(d.rows.map((r,i)=>[`${r.config}|${r.dataset}`,{r,excelRow:i+2}]));
const h=['实验','阈值','k',...d.labels.flatMap(x=>[`${x}\n准确率`,`${x}\n平均介入次数`,`${x}\nTeacher token占比`,`${x}\n平均输出token`])];
for(const s of [main,detail,notes])s.showGridLines=false;
main.tabColor='#25445C';detail.tabColor='#6F8CA4';notes.tabColor='#A8ADB3';
main.getRange('A2').values=[['数学、推理与代码基准总表']];
main.getRange('A3').values=[['27种配置 × 8个数据集；每题单次non-thinking rollout']];
main.getRange('A4').values=[['按阈值固定、k递增排列。停止规则存在差异，详见“口径与来源”。']];
main.getRange('A6:AI6').values=[h];
main.getRange('A7:C33').values=d.configs.map(g=>[label(g),...params(g)]);
for(let i=0;i<d.configs.length;i++){
 const cells=[];
 for(const ds of d.datasets){const row=map.get(`${d.configs[i]}|${ds}`).excelRow;cells.push(...['G','H','I','J'].map(c=>`='可筛选明细'!${c}${row}`));}
 main.getRange(`D${i+7}:AI${i+7}`).formulas=[cells];
}
main.getRange('A2:AI33').format.font={name:'Arial',size:10,color:'#172B3A'};
main.getRange('A2').format.font={name:'Arial',size:16,bold:true};
main.getRange('A3:A4').format.font={name:'Arial',size:10,color:'#566573'};
main.getRange('A2:AI33').format.verticalAlignment='center';
main.getRange('A6:AI33').format.columnWidth=16;
main.getRange('A6:A33').format.columnWidth=19;
main.getRange('B6:C33').format.columnWidth=8;
main.getRange('A7:AI33').format.rowHeight=25;
main.getRange('D7:AI33').setNumberFormat('0.00');
for(let j=0;j<8;j++){
 for(const offset of [0,2])main.getRange(`${col(3+j*4+offset)}7:${col(3+j*4+offset)}33`).setNumberFormat('0.00%');
 const g=main.getRange(`${col(3+j*4)}6:${col(6+j*4)}6`);
 g.format.fill=j<3?'#25445C':'#385B6A';
 g.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
}
main.getRange('A6:C6').format.fill='#25445C';
main.getRange('A6:C6').format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
main.getRange('A6:AI6').format.wrapText=true;
main.getRange('A6:AI6').format.rowHeight=48;
main.getRange('A6:AI6').format.horizontalAlignment='center';
main.getRange('B7:AI33').format.horizontalAlignment='right';
for(let r=9;r<=33;r+=5){
 main.getRange(`A${r}:AI${r+4}`).format.fill=((r-9)/5)%2===0?'#F1F5F8':'#FFFFFF';
 main.getRange(`A${r}:AI${r}`).format.borders={top:{style:'thin',color:'#B9C7D0'}};
}
main.getRange('A7:AI8').format.fill='#E5EDF3';
main.freezePanes.freezeRows(6);main.freezePanes.freezeColumns(3);

const dh=['实验','阈值','k','数据集','题数','正确题数','准确率','平均介入次数','Teacher token占比','平均输出token','生成超时题数','停止规则','来源'];
detail.getRange('A1:M217').values=[dh,...d.rows.map(r=>[label(r.config),...params(r.config),d.labels[d.datasets.indexOf(r.dataset)],r.n,r.correct,r.acc,r.mean_interventions,r.teacher_token_fraction,r.mean_output_tokens,r.generation_timeouts,r.stop_policy,r.source])];
detail.getRange('A1:M217').format.font={name:'Arial',size:10,color:'#172B3A'};
detail.getRange('A1:M217').format.verticalAlignment='center';
detail.getRange('A1:M217').format.rowHeight=22;
detail.getRange('A1:M217').format.columnWidth=16;
detail.getRange('A1:A217').format.columnWidth=19;detail.getRange('B1:C217').format.columnWidth=8;
detail.getRange('D1:D217').format.columnWidth=20;
detail.getRange('L1:L217').format.columnWidth=65;detail.getRange('M1:M217').format.columnWidth=48;
detail.getRange('G2:G217').setNumberFormat('0.00%');detail.getRange('I2:I217').setNumberFormat('0.00%');
detail.getRange('H2:H217').setNumberFormat('0.00');detail.getRange('J2:J217').setNumberFormat('0.00');
detail.getRange('E2:K217').format.horizontalAlignment='right';
const table=detail.tables.add('A1:M217',true,'AllExperimentRecords');table.showFilterButton=true;
detail.getRange('A1:M1').format.fill='#25445C';detail.getRange('A1:M1').format.font={name:'Arial',size:10,color:'#FFFFFF',bold:true};
detail.getRange('A1:M1').format.wrapText=true;detail.getRange('A1:M1').format.rowHeight=38;detail.getRange('A1:M1').format.horizontalAlignment='center';
detail.freezePanes.freezeRows(1);detail.freezePanes.freezeColumns(4);

const noteRows=[
 ['项目','说明'],
 ['模型与采样','Student：Qwen3-1.7B；Teacher：Qwen3-14B。单次non-thinking；temperature=1、top_p=1。'],
 ['无Luna网格','k=1、4、8、16、32；正有符号差 logprob(Student)−logprob(Teacher) 严格大于阈值。不限介入次数；Teacher首token禁止原Student token。'],
 ['平均介入次数','每题的实际Teacher接管次数，含零介入题目。Teacher-only为直接生成，介入次数记0，token占比为100%。'],
 ['Teacher token占比','Teacher保留token总数 ÷ 实际输出token总数。按token总量加权；不含prompt或被丢弃的Student草稿。'],
 ['平均输出长度','保留的response_token_ids长度 ÷ 题数，包含超时题目的已保留部分。'],
 ['数学/推理停止规则','历史基线无900秒截止。数学网格19组未新增截止；k16/阈值4中373条旧输出、1881条启用900秒；k32五组全部启用900秒。'],
 ['代码停止规则','全部27组启用900秒累计生成截止，包含RPC等待、排除未分配排队与作业停机。生成超时保留部分输出并计错，分母不减少。'],
 ['跨实验比较限制','停止规则不完全一致；数学历史直接生成基线和代码64-token分段基线的生成组织方式不同。总表不提供跨数据集总准确率。'],
 ['代码执行评分','使用官方CRUXEval、EvalPlus增强测试、LiveCodeBench执行测试。代码执行超时及进程崩溃计错；“生成超时题数”不包含评分执行超时。'],
 ['数学/推理题集','MATH500：500题；OlympiadBench：OE_TO_maths_en_COMP 674题；BBH：27子任务各40题，共1080题。'],
 ['代码题集','CRUXEval-O/I各800题；HumanEval+ v0.1.10 164题；MBPP+ v0.2.0 378题；LiveCodeBench v6增量175题（2025-01-04至04-06）。'],
 ['M：数学网格来源','nojudge_grid_20260912最终FINAL_COMPARISON.csv；本次只读汇总原始completed文件，补充长度、超时及token来源，并核对已有占比。'],
 ['C：代码网格来源','code_reasoning_grid_20260913最终FINAL_COMPARISON.csv，27组、62559条评分、135条数据集汇总。'],
 ['H：历史基线来源','hard_three_20260911/artifacts/summary.json和原始Student/Teacher输出；仅纳入OlympiadBench和BBH，排除AIME。'],
 ['S/T：MATH500基线来源','Student：math500_three_20260911/results/summary.json。Teacher：threshold3_mathteacher_20260911/summary.json中的直接生成结果及原始输出。'],
 ['覆盖范围','27种配置 × 8个数据集 = 216条汇总记录；共123417条评测输出。先列两组基线，其后按阈值、k递增。']
];
notes.getRange(`A1:B${noteRows.length}`).values=noteRows;
notes.getRange(`A1:B${noteRows.length}`).format.font={name:'Arial',size:10,color:'#172B3A'};
notes.getRange(`A1:B${noteRows.length}`).format.verticalAlignment='center';
notes.getRange(`A1:A${noteRows.length}`).format.columnWidth=26;
notes.getRange(`B1:B${noteRows.length}`).format.columnWidth=105;
notes.getRange(`A1:B${noteRows.length}`).format.wrapText=true;
notes.getRange(`A2:B${noteRows.length}`).format.rowHeight=46;
notes.getRange('A1:B1').format.fill='#25445C';notes.getRange('A1:B1').format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
notes.getRange('A1:B1').format.rowHeight=27;
wb.recalculate();
const check=await wb.inspect({kind:'region',sheetId:main.sheetId,range:'A7:G10',maxChars:1800});
console.log(check.ndjson);
for(let i=0;i<d.configs.length;i++)for(let j=0;j<8;j++){
 const r=map.get(`${d.configs[i]}|${d.datasets[j]}`).r;
 const expected=[r.acc,r.mean_interventions,r.teacher_token_fraction,r.mean_output_tokens];
 const actual=main.getRange(`${col(3+j*4)}${i+7}:${col(6+j*4)}${i+7}`).values[0];
 actual.forEach((v,k)=>{if(typeof v!=='number'||Math.abs(v-expected[k])>1e-9)throw new Error(`Formula mismatch ${i},${j},${k}: ${v}`);});
}
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
for(const [sheet,range,file] of [[main,'A1:K14','preview_main.png'],[main,'X6:AI12','preview_code.png'],[detail,'A1:K12','preview_detail.png'],[notes,'A1:B17','preview_notes.png']]){
 const blob=await wb.render({sheetName:sheet.name,range,scale:1.5,format:'png'});
 await fs.writeFile(path.join(out,file),new Uint8Array(await blob.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'全部实验_数学推理与代码.xlsx'));
console.log('EXPORTED: 864 numeric summary formulas verified against216 records; all3 sheets rendered.');
