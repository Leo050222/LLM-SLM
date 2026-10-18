from common import *
def draw(metric,name,datasets=None):
 overview=datasets is None
 datasets=list(DS) if overview else datasets
 fig,axs=plt.subplots(3 if overview else 2,3 if overview else 2,figsize=(13,12) if overview else (8.5,8.5))
 fig.subplots_adjust(left=.06,right=.95,top=.94,bottom=.08,wspace=.45,hspace=.64)
 for i,(ax,d) in enumerate(zip(axs.flat,datasets)):
  panel(ax,d,i);arr=np.array([r[metric] for r in grid(d)]).reshape(5,5)
  if metric=='mean_output_tokens':arr=arr/DATA['student_only',d][metric]
  im=ax.imshow(arr,cmap='Blues',aspect='auto',vmin=0 if metric=='mean_interventions' else None)
  cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.035);cb.ax.tick_params(labelsize=8)
  cb.set_label({'accuracy_pct':'Mean accuracy (%)','mean_interventions':'Interventions / output','mean_output_tokens':'Length / Student length'}[metric],fontsize=8)
  for y in range(5):
   for x in range(5):
    val=arr[y,x];ax.text(x,y,f'{val:.2f}',ha='center',va='center',fontsize=9,color='white' if im.norm(val)>.6 else '#142d41')
    if metric=='accuracy_pct' and np.isclose(val,arr.max(),rtol=0,atol=1e-9):ax.add_patch(Rectangle((x-.47,y-.47),.94,.94,fill=False,edgecolor='#ed9500',lw=2.1))
  ax.set_xticks(range(5),KS);ax.set_yticks(range(5),[f'{t:g}' for t in TS]);ax.set_xlabel('k');ax.set_ylabel('Threshold')
  s,t=DATA['student_only',d],DATA['teacher_only',d]
  if metric=='accuracy_pct':note=f'Student {s[metric]:.2f}%   |   Teacher {t[metric]:.2f}%'
  elif metric=='mean_output_tokens':note=f'Student 1.00× ({s[metric]:.1f} tokens)  |  Teacher {t[metric]/s[metric]:.2f}×'
  else:note='Student / Teacher-only: 0 interventions'
  ax.text(.5,-.32,note,transform=ax.transAxes,ha='center',fontsize=8)
 if overview:axs.flat[-1].axis('off')
 if overview:axs.flat[-1].text(.02,.9,'Top row: math / reasoning\nOther rows: code\n\nEach panel has its own color scale.\nEight rollouts per question.\n'+('Orange outlines: all observed maxima.\nNo significance claim.' if metric=='accuracy_pct' else ''),transform=axs.flat[-1].transAxes,fontsize=10,linespacing=1.8,va='top')
 save(fig,name)

 if overview:
  for part in range(2):draw(metric,name+f'_print_part{part+1}',list(DS)[part*4:(part+1)*4])
