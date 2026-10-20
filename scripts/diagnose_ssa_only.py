"""Prespecified SSA ranks and one-component deletions on original observations."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from ssa_task_srd.data import load_series,fit_scale,observed_input,make_windows
from ssa_task_srd.experiment import cached_components,metrics
from ssa_task_srd.ssa_diagnostics import fixed_variants


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    protocol=json.loads((root/'configs/ssa_only_protocol.json').read_text(encoding='utf-8'))
    config=json.loads((root/protocol['base_config']).read_text(encoding='utf-8'))
    assert config['repair_backend']=='none' and config['input_condition']=='reference'
    assert config['noise_std']==0 and config['outlier_rate']==0 and config['evaluation_splits']==['val']
    values,times,audit=load_series(config,root)
    train_end=int(.7*len(values));mean,std=fit_scale(values,train_end);reference=(values-mean)/std
    observed,mask=observed_input(reference,config)
    np.testing.assert_array_equal(observed,reference);assert not mask.any()
    windows,boundaries=make_windows(reference,observed,mask,config)
    assert set(windows)=={'train','val'}
    components,cache=cached_components(root,windows,dict(config,rank=protocol['full_decomposition_rank']))
    variants={split:fixed_variants(w['x'],components[split],protocol['static_ranks'],protocol['ablation_rank'])
        for split,w in windows.items()}
    controls={}
    for split in windows:
        for name in ['all_components_identity',f'ssa_rank_{protocol["full_decomposition_rank"]}']:
            delta=float(np.abs(variants[split][name]-windows[split]['x']).max())
            assert delta<1e-10;controls[f'{split}_{name}_max_input_difference']=delta
        np.testing.assert_array_equal(variants[split]['drop_residual_after_6'],variants[split]['ssa_rank_6'])
    rows=[];per_window=[];predictions={}
    for name in variants['train']:
        model=Ridge(alpha=config['ridge_alpha'],solver=config['ridge_solver']).fit(variants['train'][name],windows['train']['y'])
        prediction=model.predict(variants['val'][name]);predictions[name]=prediction
        score=metrics(prediction,windows['val']['y'],std)
        delta=variants['val'][name]-windows['val']['x']
        rows.append({'method':name,'forecast_mae_original':score['mae'],'forecast_rmse_original':score['rmse'],
            'input_change_rms_z':float(np.sqrt(np.square(delta).mean())),
            'mean_abs_first_difference_z':float(np.abs(np.diff(variants['val'][name],axis=1)).mean())})
        for origin,pred,target in zip(windows['val']['origins'],prediction,windows['val']['y']):
            m=metrics(pred,target,std)
            per_window.append({'method':name,'origin':int(origin),'forecast_mae_original':m['mae'],'forecast_rmse_original':m['rmse']})
        np.savez_compressed(out/f'predictions_{name}.npz',prediction_z=prediction,target_z=windows['val']['y'],origins=windows['val']['origins'])
    for name in ['all_components_identity','ssa_rank_144']:
        error=float(np.abs(predictions[name]-predictions['raw']).max()*std)
        assert error<1e-6;controls[name+'_max_prediction_difference_original']=error
    frame=pd.DataFrame(rows)
    baseline=float(frame.loc[frame.method=='raw','forecast_rmse_original'].iloc[0])
    frame['rmse_minus_raw']=frame.forecast_rmse_original-baseline
    frame.to_csv(out/'static_metrics.csv',index=False)
    pd.DataFrame(per_window).to_csv(out/'per_window_metrics.csv',index=False)
    numerical=[]
    for solver,tol in [('lsqr',1e-4),('lsqr',1e-12),('cholesky',1e-4)]:
        pair={}
        for name in ['raw','ssa_rank_144']:
            model=Ridge(alpha=config['ridge_alpha'],solver=solver,tol=tol).fit(variants['train'][name],windows['train']['y'])
            pair[name]=model.predict(variants['val'][name])
        numerical.append({'solver':solver,'tol':tol,
            'raw_rmse_original':metrics(pair['raw'],windows['val']['y'],std)['rmse'],
            'full_rank_rmse_original':metrics(pair['ssa_rank_144'],windows['val']['y'],std)['rmse'],
            'max_full_vs_raw_prediction_difference_original':float(np.abs(pair['raw']-pair['ssa_rank_144']).max()*std),
            'raw_max_prediction_difference_vs_direct':float(np.abs(pair['raw']-predictions['raw']).max()*std)})
    pd.DataFrame(numerical).to_csv(out/'numerical_audit.csv',index=False)
    audit.update({'boundaries':boundaries,'scale_mean':mean,'scale_std':std,'scaler_fit_range':[0,train_end],
        'window_counts':{k:len(v['x']) for k,v in windows.items()},'sampled_origins':{k:v['origins'].tolist() for k,v in windows.items()},
        'target_ranges':{k:[[int(o),int(o+config['horizon'])] for o in v['origins']] for k,v in windows.items()},
        'artificial_corruption_applied':False,'injection_function_called':False,'repair_enabled':False,
        'test_windows_constructed':False,'controls':controls,'ssa_cache':cache,
        'meaning':'Input-change RMS is modification magnitude, not denoising error; original observations are not clean truth'})
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    np.savez_compressed(out/'inputs.npz',train=windows['train']['x'],val=windows['val']['x'],components_val=components['val'])
    plot(out,frame,variants['val'],windows['val']['x'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    (out/'metadata.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':commit,'elapsed_seconds':time.perf_counter()-started,'no_test_evaluation':True},indent=2),encoding='utf-8')
    print(frame.to_string(index=False));print('SSA-only static controls passed; no corruption or repair used.')


def plot(out,frame,variants,original):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline=frame.loc[frame.method=='raw','forecast_rmse_original'].iloc[0]
    ranks=frame[frame.method.str.startswith('ssa_rank_')].copy()
    ranks['rank']=ranks.method.str.rsplit('_',n=1).str[-1].astype(int)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.plot(ranks['rank'],ranks.forecast_rmse_original,marker='o',label='Fixed-rank SSA + Ridge')
    ax.axhline(baseline,color='black',ls='--',label='Raw + Ridge')
    ax.set_xscale('log',base=2);ax.set_xticks(ranks['rank'],ranks['rank'].astype(str),rotation=25)
    ax.set_xlabel('Retained SSA components');ax.set_ylabel('Validation forecast RMSE')
    ax.set_title('Original observations; no added corruption or SRD');ax.legend()
    fig.savefig(out/'rank_sweep.png',dpi=150);plt.close(fig)
    dropped=frame[frame.method.str.startswith('drop_')]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    ax.bar(dropped.method,dropped.rmse_minus_raw,color=['tab:green' if v<0 else 'tab:red' for v in dropped.rmse_minus_raw])
    ax.axhline(0,color='black',lw=1);ax.tick_params(axis='x',labelrotation=25)
    ax.set_ylabel('Validation RMSE minus raw (negative is better)')
    ax.set_title('Remove one group; retain every other group, including residual')
    fig.savefig(out/'component_deletions.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.plot(original[0],label='Original observation',color='black',lw=1)
    for name in ['ssa_rank_2','ssa_rank_6','ssa_rank_24']:
        ax.plot(variants[name][0],label=name,lw=.8,alpha=.8)
    ax.set_xlabel('Historical point');ax.set_ylabel('Standardized input')
    ax.set_title('Validation history example; smoothness is not accuracy');ax.legend()
    fig.savefig(out/'ssa_histories.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
