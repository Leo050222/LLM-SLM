from common import *
def draw(zoom=False,datasets=None,suffix=""):
 datasets=list(DS) if datasets is None else datasets
 compact=len(datasets)==4
 fig,axs=plt.subplots(2,2 if compact else 4,figsize=(8.5,8) if compact else (16,8))
 fig.subplots_adjust(left=.09 if compact else .055,right=.98,bottom=.12,top=.85,wspace=.32,hspace=.48)
 for i,(ax,d) in enumerate(zip(axs.flat,datasets)):
  panel(ax,d,i);rs=grid(d);s=DATA['student_only',d];teacher=DATA['teacher_only',d]
  vals=[r['accuracy_pct'] for r in rs]+[teacher['accuracy_pct']]+([] if zoom else [s['accuracy_pct']])
  lo,hi=min(vals),max(vals);span=max(hi-lo,1);ax.set_ylim(lo-.1*span,hi+.35*span)
  ax.set_xlim(0,max(r['teacher_token_pct'] for r in rs)*1.12)
  for t,col in zip(TS,COLORS):
   rr=[DATA[f'k{k}_t{t:g}',d] for k in KS]
   ax.plot([r['teacher_token_pct'] for r in rr],[r['accuracy_pct'] for r in rr],color=col,lw=1.2,alpha=.85)
   for r,m in zip(rr,MARKERS):ax.scatter(r['teacher_token_pct'],r['accuracy_pct'],s=29,marker=m,color=col,edgecolors='white',linewidths=.4,zorder=3)
  ax.axhline(teacher['accuracy_pct'],color='#333333',ls='--',lw=1)
  if not zoom:ax.axhline(s['accuracy_pct'],color='#888888',ls=':',lw=1.2)
  else:ax.text(.98,.03,f'Student: {s["accuracy_pct"]:.2f}% (below axis)',transform=ax.transAxes,ha='right',fontsize=8,color='#777777')
  best=max(r['accuracy_pct'] for r in rs)
  labels=[DATA['k1_t2',d]]+[r for r in rs if np.isclose(r['accuracy_pct'],best,rtol=0,atol=1e-9) and r['config']!='k1_t2']
  for j,r in enumerate(labels):
   x,y=r['teacher_token_pct'],r['accuracy_pct'];ax.scatter(x,y,s=85,facecolors='none',edgecolors='#222222',linewidths=.8,zorder=4)
   ax.annotate(r['config'],xy=(x,y),xytext=(.08+j*.35,.89-j*.12),textcoords='axes fraction',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.65,color='#555555'),bbox=dict(facecolor='white',edgecolor='none',pad=.6),zorder=5)
  ax.grid(alpha=.15);ax.set_xlabel('Teacher token share (%)');ax.set_ylabel('Mean accuracy (%)')
 handles=[Line2D([],[],color=c,label=f't = {t:g}') for t,c in zip(TS,COLORS)]
 handles += [Line2D([],[],color='#444444',marker=m,ls='',label=f'k = {k}') for k,m in zip(KS,MARKERS)]
 handles += ([] if zoom else [Line2D([],[],color='#888888',ls=':',label='Student-only')]) + [Line2D([],[],color='#333333',ls='--',label='Teacher-only')]
 fig.legend(handles=handles,ncol=4 if compact else 6,loc='upper center',bbox_to_anchor=(.5,.995),frameon=False)
 save(fig,'accuracy_teacher_share'+('_zoom' if zoom else '')+suffix)
if __name__=='__main__':
 draw();draw(True)
 for zoom in [False,True]:
  for part in range(2):draw(zoom,list(DS)[part*4:(part+1)*4],f'_print_part{part+1}')
