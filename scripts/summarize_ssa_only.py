"""Uniform float64/direct Ridge re-evaluation of SSA-only histories, without neural retraining."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from ssa_task_srd.data import load_series,fit_scale,observed_input,make_windows
from ssa_task_srd.experiment import cached_components,metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--static',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    run=root/args.run;static=root/args.static;out=root/args.out
    out.mkdir(parents=True,exist_ok=False)
    prediction_dir=run/('readout_'+out.name);prediction_dir.mkdir(exist_ok=False)
    config=json.loads((static/'config.json').read_text(encoding='utf-8'))
    static_audit=json.loads((static/'data_audit.json').read_text(encoding='utf-8'))
    source_manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if len(source_manifest)!=6 or any(m['exit_code'] for m in source_manifest):raise RuntimeError('SSA-only runs incomplete')
    values,_,data=load_series(config,root);mean,std=fit_scale(values,int(.7*len(values)))
    reference=(values-mean)/std;observed,mask=observed_input(reference,config)
    windows,boundaries=make_windows(reference,observed,mask,config)
    assert set(windows)=={'train','val'} and not mask.any() and data['sha256']==static_audit['sha256']
    components,_=cached_components(root,windows,config)
    fixed={'raw':{k:w['x'] for k,w in windows.items()},
        'ssa':{k:c[:,:-1,:].sum(axis=1) for k,c in components.items()}}
    static_scores=pd.read_csv(static/'static_metrics.csv').set_index('method')
    rows=[];readout_changes=[];audits=[]
    for item in source_manifest:
        folder=run/item['directory'];mode=item['training_mode'];seed=item['training_seed']
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        audit=json.loads((folder/'data_audit.json').read_text(encoding='utf-8'))
        t=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        assert cfg['noise_std']==0 and cfg['outlier_rate']==0 and cfg['input_condition']=='reference'
        assert not meta['artificial_corruption_applied'] and not meta['repair_enabled'] and not meta['original_author_java_called']
        assert not meta['test_windows_constructed'] and audit['sampled_origins']==static_audit['sampled_origins']
        assert set(audit['window_counts'])=={'train','val'}
        with np.load(folder/'perturbation.npz') as z:
            np.testing.assert_array_equal(z['reference_z'],reference)
            np.testing.assert_array_equal(z['observed_z'],reference);assert not z['outlier_mask'].any()
        with np.load(folder/'histories_task_ssa.npz') as z:
            learned={k:z[k].astype(np.float64) for k in windows}
        variants=dict(fixed,task_ssa=learned)
        old=pd.read_csv(folder/'metrics.csv').set_index('method')
        for name,inputs in variants.items():
            model=Ridge(alpha=config['ridge_alpha'],solver='cholesky').fit(inputs['train'],windows['train']['y'])
            prediction=model.predict(inputs['val']);score=metrics(prediction,windows['val']['y'],std)
            delta=inputs['val']-windows['val']['reference']
            rows.append({'training_mode':mode,'training_seed':seed,'method':name,
                'forecast_mae_original':score['mae'],'forecast_rmse_original':score['rmse'],
                'input_change_rms_z':float(np.sqrt(np.square(delta).mean()))})
            readout_changes.append({'training_mode':mode,'training_seed':seed,'method':name,
                'initial_solver':meta.get('ridge_solver','lsqr'),'initial_rmse':float(old.loc[name,'forecast_rmse_original']),
                'direct_float64_rmse':score['rmse'],'direct_minus_initial':score['rmse']-float(old.loc[name,'forecast_rmse_original'])})
            if name in ['raw','ssa']:
                label='raw' if name=='raw' else 'ssa_rank_6'
                np.testing.assert_allclose(score['rmse'],static_scores.loc[label,'forecast_rmse_original'],atol=1e-6,rtol=0)
            np.savez_compressed(prediction_dir/f'{item["directory"]}_{name}.npz',
                prediction_z=prediction,target_z=windows['val']['y'],origins=windows['val']['origins'])
        previous=root/'runs/collaborative_initial'/f'reference_{mode}_seed{seed}'/'training.json'
        history_check=None
        if previous.exists():
            original=json.loads(previous.read_text(encoding='utf-8'))
            history_check=all(t[key]==original[key] for key in ['pretrained_proxy_sha256','final_proxy_sha256','final_head_sha256','batch_order_sha256'])
            assert history_check,'Removing SRD changed neural training unexpectedly'
        audits.append({'training_mode':mode,'training_seed':seed,'training_code_commit':meta['git_commit'],
            'pretrained_proxy_sha256':t['pretrained_proxy_sha256'],'initial_head_sha256':t['initial_trainable_selector_sha256'],
            'batch_order_sha256':t['batch_order_sha256'],'neural_weights_unchanged_by_readout_correction':True,
            'matches_previous_reference_neural_training':history_check,
            'no_added_corruption':True,'no_repair':True,'test_evaluation':False,
            'learning_seconds':meta['timings']['learning']['seconds']})
    for seed in [42,43,44]:
        pair=[a for a in audits if a['training_seed']==seed]
        for key in ['pretrained_proxy_sha256','initial_head_sha256','batch_order_sha256']:
            assert len({a[key] for a in pair})==1
    frame=pd.DataFrame(rows);frame.to_csv(out/'learned_metrics.csv',index=False)
    summary=frame.groupby(['training_mode','method'])[['forecast_mae_original','forecast_rmse_original','input_change_rms_z']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(out/'learned_summary.csv',index=False)
    pd.DataFrame(readout_changes).to_csv(out/'readout_solver_comparison.csv',index=False)
    for name in ['static_metrics.csv','per_window_metrics.csv','numerical_audit.csv','rank_sweep.png','component_deletions.png','ssa_histories.png']:
        shutil.copyfile(static/name,out/name)
    (out/'static_data_audit.json').write_text(json.dumps(static_audit,indent=2),encoding='utf-8')
    audit={'status':'All checks passed: original inputs, no injection/repair/test, paired initialization, stable identity readout',
        'readout':'Uniform float64 Ridge alpha=1, solver=cholesky; neural histories/weights unchanged',
        'evaluation_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'protocol':json.loads((root/'configs/ssa_only_protocol.json').read_text(encoding='utf-8')),
        'model_runs':audits,'initial_failed_diagnostic_retained':str((run/'static_diagnostics').relative_to(root)),
        'scope':'Validation exploration; not clean-ground-truth denoising evaluation or confirmed tuned rank'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))
    per=pd.read_csv(static/'per_window_metrics.csv')
    pivot=per.pivot(index='origin',columns='method',values='forecast_rmse_original')
    print('drop_C4 improves per-window RMSE in',int((pivot.drop_C4<pivot.raw).sum()),'of',len(pivot),'validation windows')
    print('All SSA-only audits passed; only readout models refitted, no neural retraining for correction.')


if __name__=='__main__':main()
