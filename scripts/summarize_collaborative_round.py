"""Audit and summarize the matched alternating-training validation experiment."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    source=root/args.run;out=root/args.out;out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if len(manifest)!=12 or any(m['exit_code'] for m in manifest):raise RuntimeError('Incomplete paired matrix')
    rows=[];diagnostics=[];audits=[];epochs=[]
    for entry in manifest:
        folder=source/entry['directory'];mode=entry['training_mode'];condition=entry['condition'];seed=entry['seed']
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        t=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        data=json.loads((folder/'data_audit.json').read_text(encoding='utf-8'))
        table=pd.read_csv(folder/'metrics.csv')
        assert set(table.split)=={'val'} and not meta['test_windows_constructed']
        assert set(data['window_counts'])=={'train','val'}
        assert meta['repair_backend']=='java' and meta['original_java_parity_verified']
        assert t['training_mode']==mode and t['selector_head_weights_updated'] and t['coupled_checkpoint_restored']
        assert t['selector_optimizer_steps']==75 and t['proxy_optimizer_steps']==(75 if mode=='collaborative' else 0)
        assert t['proxy_weights_unchanged']==(mode=='paired_frozen')
        assert t['shared_features_updated']==(mode=='collaborative')
        assert 1<=t['best_epoch']<=15
        phase=t['first_batch_phase_audit']
        assert phase['selector_phase']['proxy_unchanged'] and phase['selector_phase']['head_updated']
        assert phase['proxy_phase']['head_unchanged'] and phase['proxy_phase']['executed']==(mode=='collaborative')
        assert phase['proxy_phase']['proxy_updated']==(mode=='collaborative')
        with np.load(folder/'perturbation.npz') as z:
            boundary=data['split_boundaries']['val_end']
            input_hash=hashlib.sha256(z['observed_z'][:boundary].tobytes()).hexdigest()
        for r in table.to_dict('records'):rows.append(dict(r,condition=condition,training_mode=mode,training_seed=seed))
        for r in t['epoch_records']:epochs.append(dict(r,condition=condition,training_mode=mode,training_seed=seed))
        w0=t['initial_weight_summary']['val'];w1=t['final_weight_summary']['val']
        diagnostics.append({'condition':condition,'training_mode':mode,'training_seed':seed,
            'best_epoch':t['best_epoch'],'initial_reconstructed_proxy_mse_z':t['initial_selector_proxy_mse_z']['val'],
            'final_reconstructed_proxy_mse_z':t['proxy_diagnostics']['val']['task_ssa_mse_z'],
            'final_reference_proxy_mse_z':t['proxy_diagnostics']['val']['reference_mse_z'],
            'final_observed_proxy_mse_z':t['proxy_diagnostics']['val']['observed_mse_z'],
            'initial_weight_mean':w0['mean'],'final_weight_mean':w1['mean'],
            'initial_component_std':w0['mean_within_window_std'],'final_component_std':w1['mean_within_window_std'],
            'head_steps':t['selector_optimizer_steps'],'proxy_steps':t['proxy_optimizer_steps'],
            'learning_seconds':meta['timings']['learning']['seconds'],'total_run_seconds':meta['elapsed_seconds_total']})
        audits.append({'condition':condition,'training_mode':mode,'training_seed':seed,
            'pretrained_proxy_sha256':t['pretrained_proxy_sha256'],'initial_head_sha256':t['initial_trainable_selector_sha256'],
            'batch_order_sha256':t['batch_order_sha256'],'input_sha256':input_hash,'origins':data['sampled_origins'],
            'code_commit':meta['git_commit'],'author_jar_sha256':meta['java_srd_provenance']['author_jar_sha256'],
            'best_epoch':t['best_epoch'],'checkpoint_restored':t['coupled_checkpoint_restored'],
            'final_proxy_sha256':t['final_proxy_sha256'],'final_head_sha256':t['final_head_sha256'],
            'first_batch_phase_audit':phase,'final_weight_mean_by_component':w1['mean_by_component']})
    for seed in protocol['training_seeds']:
        for condition in protocol['conditions']:
            pair=[a for a in audits if a['training_seed']==seed and a['condition']==condition]
            for key in ['pretrained_proxy_sha256','initial_head_sha256','batch_order_sha256','input_sha256','origins','author_jar_sha256']:
                assert len({json.dumps(a[key],sort_keys=True) for a in pair})==1,f'Matched control failed: {key}'
    for condition in protocol['conditions']:
        assert len({a['input_sha256'] for a in audits if a['condition']==condition})==1
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    score_columns=['forecast_mae_original','forecast_rmse_original','history_rmse_z']
    summary=frame.groupby(['condition','training_mode','method'])[score_columns].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(out/'summary.csv',index=False)
    for _,group in frame[frame.method.isin(['raw','ssa','srd_java','ssa_srd_java'])].groupby(['condition','method']):
        assert np.ptp(group.forecast_rmse_original.to_numpy())<1e-8
    contrasts=[]
    for condition in protocol['conditions']:
        for method in ['task_ssa','task_ssa_srd_java']:
            group=frame[(frame.condition==condition)&(frame.method==method)]
            pivot=group.pivot(index='training_seed',columns='training_mode',values='forecast_rmse_original')
            diff=pivot['collaborative']-pivot['paired_frozen']
            contrasts.append({'condition':condition,'method':method,'comparison':'collaborative minus matched frozen',
                'paired_rmse_mean':float(diff.mean()),'paired_rmse_sample_std':float(diff.std(ddof=1)),
                'wins_out_of_3':int((diff<0).sum()),'paired_differences':json.dumps(diff.tolist())})
    increments=[]
    for (condition,mode,seed),group in frame.groupby(['condition','training_mode','training_seed']):
        scores=group.set_index('method').forecast_rmse_original
        increments.append({'condition':condition,'training_mode':mode,'training_seed':seed,
            'combo_minus_srd':float(scores['task_ssa_srd_java']-scores['srd_java']),
            'combo_minus_raw':float(scores['task_ssa_srd_java']-scores['raw']),
            'task_ssa_minus_raw':float(scores['task_ssa']-scores['raw'])})
    pd.DataFrame(contrasts).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(increments).to_csv(out/'plugin_increments.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv',index=False)
    pd.DataFrame(epochs).to_csv(out/'epochs.csv',index=False)
    audit={'status':'Passed: paired initialization/batches/inputs, phase isolation, shared encoder updates, checkpoint restoration, original Java, validation-only',
        'protocol':protocol,'run_audits':audits,
        'limits':['Three training seeds with one fixed corruption realization','36 train/12 val windows',
            'Validation proxy chooses checkpoint; independent Ridge does not','No full EEG TASK reproduction or novelty claim']}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    plot(out,frame,pd.DataFrame(diagnostics),pd.DataFrame(epochs))
    print(summary[['condition','training_mode','method','forecast_rmse_original_mean','forecast_rmse_original_std']].to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(pd.DataFrame(diagnostics).to_string(index=False))
    print('All 12 matched-run audits passed.')


def plot(out,frame,diagnostics,epochs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    series=[('raw','paired_frozen','Raw'),('srd_java','paired_frozen','Author SRD'),
        ('task_ssa','paired_frozen','Frozen SSA'),('task_ssa','collaborative','Alternating SSA'),
        ('task_ssa_srd_java','paired_frozen','Frozen SSA\n+ SRD'),('task_ssa_srd_java','collaborative','Alternating SSA\n+ SRD')]
    for ax,condition in zip(axes,['reference','mixed']):
        means=[];stds=[]
        for method,mode,_ in series:
            values=frame[(frame.condition==condition)&(frame.training_mode==mode)&(frame.method==method)].forecast_rmse_original
            means.append(values.mean());stds.append(values.std(ddof=1))
        ax.bar(range(len(series)),means,yerr=stds,capsize=3)
        ax.set_xticks(range(len(series)),[s[2] for s in series],rotation=25,ha='right',fontsize=8)
        ax.set_ylabel('Independent Ridge validation RMSE')
        ax.set_title('No added corruption' if condition=='reference' else 'Fixed mixed corruption')
    fig.suptitle('Matched frozen vs alternating; 3 training seeds; mean +/- sample SD')
    fig.savefig(out/'independent_forecast.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,condition in zip(axes,['reference','mixed']):
        for mode,label in [('paired_frozen','Frozen'),('collaborative','Alternating')]:
            subset=epochs[(epochs.condition==condition)&(epochs.training_mode==mode)]
            aggregate=subset.groupby('epoch').val_reconstructed_proxy_mse_z.agg(['mean','std'])
            ax.plot(aggregate.index,aggregate['mean'],label=label)
            ax.fill_between(aggregate.index,aggregate['mean']-aggregate['std'],aggregate['mean']+aggregate['std'],alpha=.15)
        ax.set_xlabel('Epoch');ax.set_ylabel('Reconstructed validation proxy MSE (z units)')
        ax.set_title(condition);ax.legend()
    fig.suptitle('Proxy performance is distinct from independent forecasting performance')
    fig.savefig(out/'proxy_training.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
