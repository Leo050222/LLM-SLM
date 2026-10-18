from pathlib import Path
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/opd-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / 'rollout_shortening_accuracy_20260907'
rows = list(csv.DictReader((root / 'buckets.csv').open()))
summary = json.loads((root / 'summary.json').read_text())
for h in (32, 64, 128, 256):
    for scope in ('all', 'clean'):
        select = lambda kind: [r for r in rows if int(r['horizon']) == h and r['scope'] == scope and r['grouping'] == kind]
        ov = {r['bucket']: r for r in select('overview')}
        assert int(ov['shorter']['rollouts']) + int(ov['not_shorter']['rollouts']) == int(ov['all']['rollouts'])
        for kind in ('relative', 'absolute', 'shorter_quintile'):
            assert sum(int(r['rollouts']) for r in select(kind)) == int(ov['shorter']['rollouts'])
            assert sum(int(r['correct']) for r in select(kind)) == int(ov['shorter']['correct'])
        if scope == 'all':
            assert abs(float(ov['all']['accuracy_pct'])/100 - summary['horizons'][str(h)]['accuracy']) < 1e-12

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
colors = ['#3566ad', '#2a947d', '#c08720', '#b54a70']
for h, color in zip((32,64,128,256), colors):
    for ax, kind, field in zip(axes, ('absolute','relative','shorter_quintile'), ('accuracy_pct','accuracy_pct','delta_vs_matched_baseline_pp')):
        rr = [r for r in rows if int(r['horizon'])==h and r['scope']=='all' and r['grouping']==kind]
        ax.plot(range(len(rr)), [float(r[field]) for r in rr], marker='o', color=color, label=f'{h}-token bridge', linewidth=2)
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set(title='More tokens saved: accuracy decreases', ylabel='Rollout accuracy (%)', xlabel='Full-trace tokens saved vs prefix baseline')
axes[0].set_xticks(range(6), ['0–64','64–128','128–256','256–512','512–1024','>1024'], rotation=30)
axes[1].set(title='Percentage saved: no universal increase', ylabel='Rollout accuracy (%)', xlabel='Full-trace reduction (%)')
axes[1].set_xticks(range(7), ['0–10','10–20','20–30','30–40','40–50','50–75','>75'], rotation=30)
axes[2].set(title='Difference from matched baseline increases', ylabel='Accuracy minus matched baseline (pp)', xlabel='Quintiles of percentage saved (shorter only)')
axes[2].set_xticks(range(5), ['Q1: least','Q2','Q3','Q4','Q5: most'])
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle('Individual after-bridge rollouts vs mean length of 8 direct rollouts', fontsize=15)
fig.text(.5, .015, '13,244 prefixes; 847,616 rollouts per horizon. Full trace includes bridge. Descriptive associations, not causal effects. >75% bins contain only 622–713 rollouts.', ha='center', fontsize=9)
fig.tight_layout(rect=(0,.055,1,.94))
fig.savefig(root/'shortening_accuracy.png', dpi=180)
fig.savefig(root/'shortening_accuracy.svg')
print('Bucket counts, correct counts, and horizon accuracy reconciled. Saved PNG and SVG.')
