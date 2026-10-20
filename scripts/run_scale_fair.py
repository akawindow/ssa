"""Scale-controlled, chronological Ridge calibration on frozen existing histories."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
import argparse
from datetime import datetime,timezone
import hashlib
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
from ssa_task_srd.readout_calibration import expanding_folds,pipeline,select_alpha


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/scale_fair_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    protocol=json.loads((root/'configs/scale_fair_protocol.json').read_text(encoding='utf-8'))
    source=root/protocol['source_run']
    config=json.loads((source/protocol['source_config']).read_text(encoding='utf-8'))
    old_audit=json.loads((source/'data_audit.json').read_text(encoding='utf-8'))
    assert config['input_condition']=='reference' and config['repair_backend']=='none'
    assert config['noise_std']==config['outlier_rate']==0 and config['evaluation_splits']==['val']
    values,_,data=load_series(config,root);mean,std=fit_scale(values,int(.7*len(values)))
    reference=(values-mean)/std;observed,mask=observed_input(reference,config)
    np.testing.assert_array_equal(observed,reference);assert not mask.any()
    windows,boundaries=make_windows(reference,observed,mask,config)
    assert set(windows)=={'train','val'} and data['sha256']==old_audit['sha256']
    assert all(w['origins'].tolist()==old_audit['origins'][k] for k,w in windows.items())
    components,cache=cached_components(root,windows,dict(config,rank=144))
    inputs={'raw':{k:w['x'].astype(np.float64) for k,w in windows.items()},
            'ssa_7':{k:c[:,:7,:].sum(axis=1) for k,c in components.items()}}
    descriptors={name:{'kind':'fixed','case':name,'seed':None} for name in inputs}
    old_fixed=pd.read_csv(source/'report/fixed_metrics.csv').set_index('method')
    previous={name:float(old_fixed.loc[name,'rmse']) for name in inputs}
    old_task=pd.read_csv(source/'report/learned_metrics.csv').set_index(['case','seed'])
    old_scalar=pd.read_csv(source/'report/scalar_controls.csv')
    source_hashes={};scales={}
    for case in protocol['task_cases']:
        for seed in protocol['training_seeds']:
            folder=source/f'{case}_seed{seed}'
            key=f'task__{case}__{seed}';scalar_key=f'scalar__{case}__{seed}'
            src_audit=json.loads((folder/'data_audit.json').read_text(encoding='utf-8'))
            src_meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
            assert not src_meta['artificial_corruption_applied'] and not src_meta['repair_enabled']
            assert not src_meta['test_windows_constructed']
            assert all(src_audit['sampled_origins'][k]==w['origins'].tolist() for k,w in windows.items())
            for filename in ['histories_task_ssa.npz','component_weights.npz']:
                path=folder/filename;source_hashes[str(path.relative_to(root))]=digest(path)
            with np.load(folder/'histories_task_ssa.npz') as z:
                assert set(z.files)=={'train','val'}
                inputs[key]={k:z[k].astype(np.float64) for k in windows}
            with np.load(folder/'component_weights.npz') as z:
                s=float(z['train'].mean());assert 0<s<1
            with np.load(folder/'predictions_task_ssa_val.npz') as z:
                np.testing.assert_array_equal(z['target_z'],windows['val']['y'])
            inputs[scalar_key]={k:w['x']*s for k,w in windows.items()}
            descriptors[key]={'kind':'task','case':case,'seed':seed}
            descriptors[scalar_key]={'kind':'scalar','case':case,'seed':seed}
            scales[scalar_key]=s
            previous[key]=float(old_task.loc[(case,seed),'rmse'])
            row=old_scalar[(old_scalar.case==case)&(old_scalar.seed==seed)&(old_scalar.method=='scalar_same_alpha')]
            previous[scalar_key]=float(row.rmse.iloc[0])
    folds=expanding_folds(len(windows['train']['x']),**{k:v for k,v in protocol['inner_folds'].items() if k in ['first_train_fraction']},
                          n_splits=protocol['inner_folds']['n_splits'])
    fold_audit=[]
    for fold,(train,val) in enumerate(folds):
        origins=windows['train']['origins']
        assert origins[train[-1]]+config['horizon']<=origins[val[0]]
        fold_audit.append({'fold':fold,'train_count':len(train),'inner_val_count':len(val),
            'train_indices':[int(train[0]),int(train[-1])+1], 'inner_val_indices':[int(val[0]),int(val[-1])+1],
            'last_training_target_end':int(origins[train[-1]]+config['horizon']),
            'first_inner_validation_origin':int(origins[val[0]])})
    rows=[];cv_rows=[];per_window=[];predictions={};normalized_raw=None;identity=[];solver_checks=[]
    (out/'predictions').mkdir()
    for name,xx in inputs.items():
        desc=descriptors[name]
        chosen,fold_rows,scores=select_alpha(xx['train'],windows['train']['y'],folds,protocol['alphas'])
        cv_rows.extend(dict(representation=name,**desc,**r) for r in fold_rows)
        model=pipeline(chosen).fit(xx['train'],windows['train']['y'])
        prediction=model.predict(xx['val']);predictions[name]=prediction
        score=metrics(prediction,windows['val']['y'],std)
        untuned=pipeline(protocol['fixed_alpha_diagnostic']).fit(xx['train'],windows['train']['y'])
        untuned_score=metrics(untuned.predict(xx['val']),windows['val']['y'],std)
        normalized={k:model['scale'].transform(a) for k,a in xx.items()}
        direct=Ridge(alpha=chosen,solver='cholesky').fit(normalized['train'],windows['train']['y'])
        solver_delta=float(np.max(np.abs(direct.predict(normalized['val'])-prediction))*std)
        assert solver_delta<1e-4,('Direct/SVD readout discrepancy',name,solver_delta)
        solver_checks.append({'representation':name,'svd_cholesky_max_prediction_difference_original':solver_delta})
        rows.append({'representation':name,**desc,'previous_unscaled_alpha1_rmse':previous[name],
            'standardized_alpha1_mae':untuned_score['mae'],'standardized_alpha1_rmse':untuned_score['rmse'],
            'chosen_alpha':chosen,'alpha_at_grid_boundary':chosen in [min(protocol['alphas']),max(protocol['alphas'])],
            'inner_cv_rmse_original':float(np.sqrt(scores[chosen])*std),
            'mae':score['mae'],'rmse':score['rmse']})
        for origin,p,y in zip(windows['val']['origins'],prediction,windows['val']['y']):
            m=metrics(p,y,std);per_window.append({'representation':name,**desc,'origin':int(origin),'mae':m['mae'],'rmse':m['rmse']})
        np.savez_compressed(out/'predictions'/f'{name}.npz',prediction_z=prediction,target_z=windows['val']['y'],
            origins=windows['val']['origins'],feature_mean=model['scale'].mean_,feature_scale=model['scale'].scale_)
        if name=='raw':
            normalized_raw=normalized;raw_alpha=chosen;raw_scores=scores
        if desc['kind']=='scalar':
            transform_delta=max(float(np.max(np.abs(normalized[k]-normalized_raw[k]))) for k in windows)
            prediction_delta=float(np.max(np.abs(prediction-predictions['raw']))*std)
            score_delta=max(abs(scores[a]-raw_scores[a]) for a in scores)
            assert chosen==raw_alpha and transform_delta<1e-10 and prediction_delta<1e-5 and score_delta<1e-9
            identity.append({'representation':name,'constant_scale':scales[name],'chosen_alpha_matches_raw':True,
                'max_standardized_input_difference':transform_delta,'max_cv_mse_difference':score_delta,
                'max_outer_prediction_difference_original':prediction_delta})
        print(f'{name}: inner-chosen alpha={chosen:g}, validation MAE={score["mae"]:.2f}, RMSE={score["rmse"]:.2f}',flush=True)
    naive=windows['val']['x'][:,-config['horizon']:]
    m=metrics(naive,windows['val']['y'],std)
    rows.append({'representation':'last_day_persistence','kind':'baseline','case':'last_day_persistence','seed':None,
                 'chosen_alpha':None,'mae':m['mae'],'rmse':m['rmse']})
    for origin,p,y in zip(windows['val']['origins'],naive,windows['val']['y']):
        m=metrics(p,y,std);per_window.append({'representation':'last_day_persistence','kind':'baseline',
            'case':'last_day_persistence','seed':None,'origin':int(origin),'mae':m['mae'],'rmse':m['rmse']})
    for path,h in source_hashes.items():assert digest(root/path)==h,'Source model outputs were changed'
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(cv_rows).to_csv(out/'inner_cv_scores.csv',index=False)
    pd.DataFrame(per_window).to_csv(out/'per_window_metrics.csv',index=False)
    pd.DataFrame(identity).to_csv(out/'scalar_identity_checks.csv',index=False)
    pd.DataFrame(solver_checks).to_csv(out/'solver_checks.csv',index=False)
    summary=frame[frame.kind.isin(['task','scalar'])].groupby(['kind','case'])[['mae','rmse']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(out/'summary.csv')
    data.update({'boundaries':boundaries,'scale_mean':mean,'scale_std':std,
        'window_counts':{k:len(v['x']) for k,v in windows.items()},'inner_folds':fold_audit,
        'origins':{k:w['origins'].tolist() for k,w in windows.items()},
        'new_neural_training':False,'artificial_corruption':False,'repair':False,'test_windows_constructed':False,
        'outer_validation_scores_used_for_ridge_alpha_selection':False,
        'upstream_models_are_frozen_and_not_refit_per_inner_fold':True,
        'upstream_checkpoint_previously_used_outer_validation':True})
    for name,value in [('protocol',protocol),('data_audit',data),('source_hashes',source_hashes)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    plot(out,frame)
    (out/'metadata.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'elapsed_seconds':time.perf_counter()-started,'readout_representations':len(inputs),
        'new_neural_training':False,'all_source_outputs_unchanged':True,
        'interpretation':protocol['interpretation'],'known_limits':protocol['known_limits']},indent=2),encoding='utf-8')
    print(summary.to_string());print('Scale-fair frozen-readout calibration completed; scalar identities passed.')


def plot(out,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    tasks=frame[frame.kind=='task'].groupby('case')[['mae','rmse']].agg(['mean','std'])
    for ax,metric in zip(axes,['mae','rmse']):
        ax.bar(tasks.index,tasks[(metric,'mean')],yerr=tasks[(metric,'std')],capsize=4)
        for name,color,style in [('raw','black','--'),('ssa_7','tab:green','-.'),('last_day_persistence','tab:red',':')]:
            ax.axhline(float(frame.loc[frame.representation==name,metric].iloc[0]),color=color,ls=style,label=name)
        ax.tick_params(axis='x',labelrotation=25);ax.set_ylabel('Validation '+metric.upper()+' (lower is better)');ax.legend(fontsize=8)
    fig.suptitle('Frozen histories; train-fitted feature scales; inner-chronological Ridge calibration')
    fig.savefig(out/'scale_fair_results.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
