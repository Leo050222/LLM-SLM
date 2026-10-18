from common import *
def draw(datasets=None,suffix=''):
 datasets=list(DS) if datasets is None else datasets
 compact=len(datasets)==4
 fig,axs=plt.subplots(2,2 if compact else 4,figsize=(8.5,8) if compact else (17,8))
 fig.subplots_adjust(left=.10 if compact else .065,right=.97,bottom=.1,top=.87,wspace=.53,hspace=.47)
 configs=['student_only','teacher_only']+[f'k{k}_t2' for k in KS]
 colors=['#e1e4e8','#a3c9e2','#215d89']
 for i,(ax,d) in enumerate(zip(axs.flat,datasets)):
  panel(ax,d,i)
  for y,c in enumerate(configs):
   r=DATA[c,d];vals=[100-r['pass_at_8_pct'],r['pass_at_8_pct']-r['pass_all_8_pct'],r['pass_all_8_pct']];left=0
   for j,(v,col) in enumerate(zip(vals,colors)):
    ax.barh(y,v,left=left,color=col,height=.66,edgecolor='white',linewidth=.5)
    if v>=8:ax.text(left+v/2,y,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if j==2 else '#263746')
    left+=v
   ax.text(103,y,f'{r["accuracy_pct"]:.2f}',va='center',fontsize=9)
  ax.text(103,-.8,'Acc. (%)',fontsize=8)
  ax.axhline(1.5,color='#999999',lw=.6)
  ax.set_yticks(range(7),['Student','Teacher']+[f'k = {k}' for k in KS]);ax.set_xlim(0,120);ax.set_xticks([0,25,50,75,100]);ax.set_ylim(6.65,-1);ax.set_xlabel('Questions (%)');ax.spines['left'].set_visible(False);ax.spines['bottom'].set_bounds(0,100);ax.tick_params(axis='y',length=0)
 fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,['0/8 correct: 100 − pass@8','1–7/8 correct: pass@8 − pass all 8','8/8 correct: pass all 8'])],ncol=1 if compact else 3,loc='upper center',frameon=False)
 save(fig,'stability_t2'+suffix)
if __name__=='__main__':
 draw()
 for part in range(2):draw(list(DS)[part*4:(part+1)*4],f'_print_part{part+1}')
