"""Post-run descriptive summaries; no model fitting or tuning."""
import argparse
import json
from pathlib import Path
import subprocess
import pandas as pd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    run=root/args.run;out=root/args.out;out.mkdir(parents=True,exist_ok=False)
    report=run/'report';learned=pd.read_csv(report/'learned_metrics.csv')
    scalar=pd.read_csv(report/'scalar_controls.csv');proxy=pd.read_csv(report/'proxy_controls.csv')
    for name,frame,columns in [('scalar_summary',scalar[scalar.method=='scalar_same_alpha'],['mae','rmse']),
                               ('proxy_summary',proxy,['proxy_task_rmse'])]:
        summary=frame.groupby('case')[columns].agg(['mean','std'])
        summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(out/f'{name}.csv')
    wide=learned.pivot(index='seed',columns='case',values='rmse');deltas=[]
    for left,right in [('full_current','coarse_current'),('full_task_head','coarse_task_head'),
                       ('coarse_task_head','coarse_current'),('full_task_head','full_current')]:
        for seed,value in (wide[left]-wide[right]).items():
            deltas.append({'left':left,'right':right,'seed':seed,'rmse_left_minus_right':value})
    pd.DataFrame(deltas).to_csv(out/'paired_factor_differences.csv',index=False)
    windows=pd.read_csv(report/'per_window_metrics.csv');raw=windows[windows['case']=='raw'].set_index('origin');wins=[]
    for (case,seed),group in windows.dropna(subset=['seed']).groupby(['case','seed']):
        group=group.set_index('origin');wins.append({'case':case,'seed':int(seed),'windows':len(group),
            'rmse_windows_below_raw':int((group.rmse<raw.rmse).sum()),'mae_windows_below_raw':int((group.mae<raw.mae).sum())})
    pd.DataFrame(wins).to_csv(out/'window_wins.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    task=learned.groupby('case').rmse.agg(['mean','std'])
    scale=scalar[scalar.method=='scalar_same_alpha'].groupby('case').rmse.agg(['mean','std'])
    fixed=pd.read_csv(report/'fixed_metrics.csv').set_index('method')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=list(range(len(task)))
    ax.errorbar(x,task['mean'],yerr=task['std'],fmt='o',capsize=4,label='Task-weighted SSA + Ridge')
    ax.errorbar(x,scale.loc[task.index,'mean'],yerr=scale.loc[task.index,'std'],fmt='s',capsize=4,label='Constant scaling only + Ridge')
    ax.axhline(fixed.loc['raw','rmse'],color='black',ls='--',label='Raw + Ridge')
    ax.axhline(fixed.loc['last_day_persistence','rmse'],color='tab:red',ls=':',label='Yesterday same time (no training)')
    ax.set_xticks(x,task.index,rotation=20);ax.set_ylabel('Validation RMSE (lower is better)')
    ax.set_title('A forecast improvement needs to exceed scaling and simple prediction controls')
    ax.legend(fontsize=9);fig.savefig(out/'scaling_comparison.png',dpi=150);plt.close(fig)
    (out/'metadata.json').write_text(json.dumps({'source_run':str(run.relative_to(root)),
        'summary_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'scope':'Post-run descriptive summary only; no new training, tuning or test evaluation'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
