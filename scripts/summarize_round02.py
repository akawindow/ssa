"""Audit the locked matrix and export small, versionable validation results."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/args.run
    out=root/args.out
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    expected=len(protocol['conditions'])*len(protocol['training_seeds'])*len(protocol['selector_modes'])
    if len(manifest)!=expected or any(m['exit_code'] for m in manifest):
        raise ValueError('Incomplete experiment matrix')
    rows=[]; diagnostics=[]; audits=[]
    for m in manifest:
        folder=source/m['directory']
        table=pd.read_csv(folder/'metrics.csv')
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        training=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        data=json.loads((folder/'data_audit.json').read_text(encoding='utf-8'))
        assert set(table['split'])=={'val'} and not meta['test_windows_constructed']
        assert set(data['window_counts'])=={'train','val'}
        assert training['proxy_weights_unchanged'] and training['selector_weights_updated']
        assert all(end<=data['split_boundaries']['val_end'] for _,end in data['target_ranges']['val'])
        with np.load(folder/'perturbation.npz') as z:
            # Compare train/validation perturbations only; no test evaluation.
            end=data['split_boundaries']['val_end']
            perturbation_hash=hashlib.sha256(z['observed_z'][:end].tobytes()).hexdigest()
            if m['condition']=='reference':
                np.testing.assert_array_equal(z['observed_z'][:end],z['reference_z'][:end])
        for record in table.to_dict('records'):
            rows.append(dict(record,condition=m['condition'],mode=m['mode'],training_seed=m['training_seed']))
        before=training['initial_weight_summary']['val']; after=training['final_weight_summary']['val']
        proxy=training['frozen_proxy_diagnostics']['val']
        diagnostics.append({'condition':m['condition'],'mode':m['mode'],'training_seed':m['training_seed'],
            'trainable_selector_parameters':training['selector_trainable_parameters'],
            'initial_weight_mean':before['mean'],'final_weight_mean':after['mean'],
            'initial_component_std':before['mean_within_window_std'],
            'final_component_std':after['mean_within_window_std'],
            'initial_selector_val_mse_z':training['initial_selector_proxy_mse_z']['val'],
            **proxy,'learning_seconds':meta['timings']['learning']['seconds'],
            'run_seconds':meta['elapsed_seconds_total']})
        audits.append({'condition':m['condition'],'mode':m['mode'],'training_seed':m['training_seed'],
            'proxy_hash':training['pretrained_proxy_sha256'],
            'head_hash':training['initial_trainable_selector_sha256'],
            'perturbation_hash':perturbation_hash,'origins':data['sampled_origins'],
            'git_commit':meta['git_commit'],'ssa_cache_hit':meta['timings']['ssa']['hit'],
            'proxy_frozen':training['proxy_weights_unchanged'],
            'final_weight_mean_by_component':after['mean_by_component']})
    for seed in protocol['training_seeds']:
        matched=[a for a in audits if a['training_seed']==seed]
        assert len({a['proxy_hash'] for a in matched})==1,'Proxy pretraining differed across controls'
        for condition in protocol['conditions']:
            pair=[a for a in matched if a['condition']==condition and a['mode']!='raw_conv']
            assert len({a['head_hash'] for a in pair})==1,'Matched heads have different initialization'
    for condition in protocol['conditions']:
        assert len({a['perturbation_hash'] for a in audits if a['condition']==condition})==1
    assert len({json.dumps(a['origins'],sort_keys=True) for a in audits})==1
    frame=pd.DataFrame(rows)
    score_columns=['forecast_mae_original','forecast_rmse_original','history_rmse_z']
    summary=frame.groupby(['condition','mode','method'])[score_columns].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns]
    # Seed repeats intentionally hold data and fixed processing constant.
    fixed=['raw','ssa','srd_prototype','ssa_srd_prototype']
    for _,group in frame[frame.method.isin(fixed)].groupby(['condition','method']):
        for column in score_columns:
            assert np.ptp(group[column].to_numpy())<1e-8,'A fixed baseline changed across controls'
    contrasts=[]
    for condition in protocol['conditions']:
        for method in ['task_ssa','task_ssa_srd_prototype']:
            group=frame[(frame.condition==condition)&(frame.method==method)]
            pivot=group.pivot(index='training_seed',columns='mode',values='forecast_rmse_original')
            for baseline in ['random_features','raw_conv']:
                diff=pivot['proxy_features']-pivot[baseline]
                contrasts.append({'condition':condition,'method':method,'comparison':'proxy_features minus '+baseline,
                    'paired_rmse_difference_mean':float(diff.mean()),'paired_difference_sample_std':float(diff.std(ddof=1)),
                    'wins_out_of_3':int((diff<0).sum()),'paired_differences':json.dumps(diff.tolist())})
    frame.to_csv(out/'metrics.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired_comparisons.csv',index=False)
    audit={'runs':expected,'evaluation':'validation only; test histories not constructed',
        'paired_seed_meaning':'Proxy initialization and training randomness; NOT different corruption realizations',
        'checks':'Passed: matched proxy/head hashes, common perturbation/history origins, frozen proxy, identical fixed baselines',
        'protocol':protocol,'run_audits':audits}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    plot(out,frame,audits)
    print(summary[['condition','mode','method','forecast_rmse_original_mean','forecast_rmse_original_std']].to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(f'Audit passed for {expected} validation-only runs. Results: {out}')


def plot(out,frame,audits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    series=[('raw','raw_conv','Raw'),('srd_prototype','raw_conv','Repair'),
        ('ssa_srd_prototype','raw_conv','SSA + repair'),
        ('task_ssa_srd_prototype','raw_conv','Conv + repair'),
        ('task_ssa_srd_prototype','proxy_features','Task features\n+ repair'),
        ('task_ssa_srd_prototype','random_features','Random features\n+ repair')]
    for ax,condition in zip(axes,['reference','mixed']):
        means=[];stds=[]
        for method,mode,_ in series:
            scores=frame[(frame.condition==condition)&(frame.method==method)&(frame['mode']==mode)].forecast_rmse_original
            means.append(scores.mean());stds.append(scores.std(ddof=1))
        ax.bar(range(len(series)),means,yerr=stds,capsize=3)
        ax.set_xticks(range(len(series)),[s[2] for s in series],rotation=30,ha='right',fontsize=8)
        ax.set_title('No added corruption' if condition=='reference' else 'Fixed mixed corruption')
        ax.set_ylabel('Validation forecast RMSE (lower is better)')
    fig.suptitle('3 training seeds; mean +/- sample SD; small validation pilot')
    fig.savefig(out/'validation_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    modes=['raw_conv','proxy_features','random_features']
    for ax,condition in zip(axes,['reference','mixed']):
        values=np.array([np.mean([a['final_weight_mean_by_component'] for a in audits
            if a['condition']==condition and a['mode']==mode],axis=0) for mode in modes])
        im=ax.imshow(values,vmin=0,vmax=1,aspect='auto',cmap='viridis')
        ax.set_yticks(range(3),modes);ax.set_xticks(range(7),['C1','C2','C3','C4','C5','C6','Residual'])
        ax.tick_params(axis='x',labelrotation=25)
        ax.set_title(condition)
        for i in range(3):
            for j in range(7): ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=7,color='black' if values[i,j]>.6 else 'white')
    fig.colorbar(im,ax=axes,label='Mean retention weight')
    fig.savefig(out/'component_weights.png',dpi=150);plt.close(fig)


if __name__=='__main__': main()
