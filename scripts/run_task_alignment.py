"""Locked single-variable audit of component grouping and TASK-style head settings."""
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
from ssa_task_srd.ssa_structure import decomposition_details,weighted_correlation,spectral_features,candidate_pair


def score_inputs(name,inputs,windows,std,alpha=1.):
    model=Ridge(alpha=alpha,solver='cholesky').fit(np.asarray(inputs['train'],dtype=np.float64),windows['train']['y'])
    pred=model.predict(np.asarray(inputs['val'],dtype=np.float64))
    m=metrics(pred,windows['val']['y'],std)
    return {'method':name,'mae':m['mae'],'rmse':m['rmse']},pred


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/task_alignment_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    protocol=json.loads((root/'configs/task_alignment_protocol.json').read_text(encoding='utf-8'))
    config=json.loads((root/protocol['base_config']).read_text(encoding='utf-8'))
    config.update(protocol['common_overrides'])
    assert config['input_condition']=='reference' and config['noise_std']==config['outlier_rate']==0
    assert config['repair_backend']=='none' and config['evaluation_splits']==['val']
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'configs').mkdir()
    values,_,audit=load_series(config,root);mean,std=fit_scale(values,int(.7*len(values)))
    reference=(values-mean)/std;observed,mask=observed_input(reference,config)
    np.testing.assert_array_equal(observed,reference);assert not mask.any()
    windows,boundaries=make_windows(reference,observed,mask,config)
    assert set(windows)=={'train','val'}
    components,cache=cached_components(root,windows,dict(config,rank=144))
    variants={'raw':{},'ssa_6':{},'ssa_7':{},'protect_candidate_C6_C7':{},'all_components_identity':{}}
    pair_protocol=json.loads((root/'configs/ssa_structure_protocol.json').read_text(encoding='utf-8'))
    pair_rows=[]
    for split,w in windows.items():
        c=components[split]
        np.testing.assert_allclose(c.sum(axis=1),w['x'],atol=1e-10,rtol=0)
        variants['raw'][split]=w['x']
        variants['ssa_6'][split]=c[:,:6,:].sum(axis=1)
        variants['ssa_7'][split]=c[:,:7,:].sum(axis=1)
        variants['all_components_identity'][split]=c.sum(axis=1)
        protect=[]
        for i,x in enumerate(w['x']):
            _,s,weights=decomposition_details(x,config['embedding'])
            cor=weighted_correlation(c[i,5:7],weights)[0,1]
            peaks=[spectral_features(a)['peak_bin'] for a in c[i,5:7]]
            protect.append(candidate_pair(s[5],s[6],cor,*peaks,pair_protocol['candidate_pair_thresholds']))
            pair_rows.append({'split':split,'origin':int(w['origins'][i]),'protect_pair':protect[-1]})
        variants['protect_candidate_C6_C7'][split]=np.where(np.array(protect)[:,None],variants['ssa_7'][split],variants['ssa_6'][split])
    fixed=[];fixed_predictions={};per_window=[]
    for name,inputs in variants.items():
        row,pred=score_inputs(name,inputs,windows,std);fixed.append(row);fixed_predictions[name]=pred
    np.testing.assert_allclose(fixed_predictions['raw'],fixed_predictions['all_components_identity'],atol=1e-6/std,rtol=0)
    persistence=windows['val']['x'][:,-config['horizon']:]
    m=metrics(persistence,windows['val']['y'],std)
    fixed.append({'method':'last_day_persistence','mae':m['mae'],'rmse':m['rmse']})
    fixed_predictions['last_day_persistence']=persistence
    for name,pred in fixed_predictions.items():
        for origin,p,y in zip(windows['val']['origins'],pred,windows['val']['y']):
            m=metrics(p,y,std);per_window.append({'case':name,'seed':None,'origin':int(origin),'mae':m['mae'],'rmse':m['rmse']})
        np.savez_compressed(out/f'predictions_{name}.npz',prediction_z=pred,target_z=windows['val']['y'])
    pd.DataFrame(fixed).to_csv(out/'fixed_metrics.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(out/'pair_choices.csv',index=False)
    audit.update({'boundaries':boundaries,'scale_mean':mean,'scale_std':std,
        'window_counts':{k:len(v['x']) for k,v in windows.items()},
        'origins':{k:v['origins'].tolist() for k,v in windows.items()},'no_added_corruption':True,
        'repair_enabled':False,'test_windows_constructed':False,'cache':cache})
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Window counts:',audit['window_counts'],flush=True)
    print(pd.DataFrame(fixed).to_string(index=False),flush=True)
    manifest=[]
    for seed in protocol['training_seeds']:
        for case in protocol['cases']:
            name=f'{case["name"]}_seed{seed}';cfg=dict(config,seed=seed,**{k:v for k,v in case.items() if k!='name'})
            path=out/'configs'/f'{name}.json';path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            print(f'Running {name}',flush=True)
            tick=time.perf_counter()
            with (out/f'{name}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-X','utf8','-m','ssa_task_srd','--config',str(path),'--out',str(out/name)],
                    cwd=root,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT)
            manifest.append({'case':case['name'],'seed':seed,'directory':name,'exit_code':result.returncode,
                             'seconds':time.perf_counter()-tick})
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            if result.returncode:raise RuntimeError(f'{name} failed; inspect log')
            print(f'Completed {name}: {manifest[-1]["seconds"]:.1f}s',flush=True)
    summarize(root,out,protocol,config,windows,std,fixed,per_window)
    sources={}
    local=Path('D:/cjx/paperstudy/eegdenoise/official')
    for name in ['base_classes.py','base_models.py','base_decompose.py','train_framework.py']:
        path=local/name
        if path.exists():sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'metadata.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'elapsed_seconds':time.perf_counter()-started,'local_TASK_source_sha256':sources},indent=2),encoding='utf-8')
    print(f'Completed locked TASK-alignment audit: {out}',flush=True)


def summarize(root,out,protocol,config,windows,std,fixed,per_window):
    rows=[];scalar_rows=[];audits=[];proxy_rows=[]
    for item in json.loads((out/'manifest.json').read_text(encoding='utf-8')):
        folder=out/item['directory'];case=item['case'];seed=item['seed']
        t=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        a=json.loads((folder/'data_audit.json').read_text(encoding='utf-8'))
        assert not meta['artificial_corruption_applied'] and not meta['repair_enabled'] and not meta['test_windows_constructed']
        assert all(a['sampled_origins'][k]==w['origins'].tolist() for k,w in windows.items())
        assert t['shared_features_updated'] and t['coupled_checkpoint_restored']
        assert t['proxy_optimizer_steps']==t['selector_optimizer_steps']
        with np.load(folder/'histories_task_ssa.npz') as z:
            inputs={k:z[k].astype(np.float64) for k in windows}
        row,pred=score_inputs(case,inputs,windows,std)
        original=pd.read_csv(folder/'metrics.csv').set_index('method').loc['task_ssa']
        np.testing.assert_allclose(row['rmse'],original.forecast_rmse_original,atol=1e-6,rtol=0)
        rows.append(dict(case=case,seed=seed,**{k:v for k,v in row.items() if k!='method'}))
        with np.load(folder/'component_weights.npz') as z:
            scale=float(z['train'].mean())
        assert 0<scale<1
        scaled={k:w['x']*scale for k,w in windows.items()}
        for name,alpha in [('scalar_same_alpha',config['ridge_alpha']),('scalar_penalty_matched',config['ridge_alpha']*scale**2)]:
            scalar,p=score_inputs(name,scaled,windows,std,alpha)
            scalar_rows.append(dict(case=case,seed=seed,scale_from_training=scale,alpha=alpha,**scalar))
            if name=='scalar_penalty_matched':
                raw=float(next(r['rmse'] for r in fixed if r['method']=='raw'))
                np.testing.assert_allclose(scalar['rmse'],raw,atol=1e-6,rtol=0)
        for origin,p,y in zip(windows['val']['origins'],pred,windows['val']['y']):
            m=metrics(p,y,std);per_window.append({'case':case,'seed':seed,'origin':int(origin),'mae':m['mae'],'rmse':m['rmse']})
        proxy_rows.append({'case':case,'seed':seed,'proxy_task_rmse':np.sqrt(t['proxy_diagnostics']['val']['task_ssa_mse_z'])*std,
                          'proxy_task_mse_z':t['proxy_diagnostics']['val']['task_ssa_mse_z']})
        if 'raw_proxy_control' in t:
            control=t['raw_proxy_control']
            assert control['initial_proxy_sha256']==t['pretrained_proxy_sha256']
            assert control['optimizer_steps']==t['proxy_optimizer_steps']
            assert control['batch_order_sha256']==t['batch_order_sha256']
            for name,key in [('raw_pretrained','initial_validation_mse_z'),('raw_extra_training','best_validation_mse_z')]:
                proxy_rows.append({'case':name,'seed':seed,'proxy_task_rmse':np.sqrt(control[key])*std,'proxy_task_mse_z':control[key]})
        audits.append({'case':case,'seed':seed,'components':cfg['rank']+1,'pretrained_proxy_sha256':t['pretrained_proxy_sha256'],
            'initial_head_sha256':t['initial_trainable_selector_sha256'],'batch_order_sha256':t['batch_order_sha256'],
            'initial_retention_mean':t['initial_weight_summary']['val']['mean'],
            'final_retention_mean':t['final_weight_summary']['val']['mean'],
            'initial_sigmoid_slope':t['initial_sigmoid_slope']['val'],'final_sigmoid_slope':t['final_sigmoid_slope']['val'],
            'final_within_window_weight_std':t['final_weight_summary']['val']['mean_within_window_std'],
            'final_head_kernel_norm':t['final_head_kernel_norm'],'head_max_norm':cfg.get('selector_max_norm'),
            'head_parameters':t['selector_step_parameters'],'proxy_parameters':t['proxy_parameters'],
            'head_steps':t['selector_optimizer_steps'],'proxy_steps':t['proxy_optimizer_steps'],'best_epoch':t['best_epoch'],
            'learning_seconds':meta['timings']['learning']['seconds'],'phase_audit':t['first_batch_phase_audit']})
        if cfg.get('selector_max_norm') is not None:assert t['final_head_kernel_norm']<=cfg['selector_max_norm']+1e-6
    for seed in protocol['training_seeds']:
        group=[a for a in audits if a['seed']==seed]
        for field in ['pretrained_proxy_sha256','batch_order_sha256','head_parameters','proxy_parameters','head_steps','proxy_steps']:
            assert len({a[field] for a in group})==1,field
        for suffix in ['current','task_head']:
            group=[a for a in audits if a['seed']==seed and a['case'].endswith(suffix)]
            assert len({a['initial_head_sha256'] for a in group})==1
    report=out/'report';report.mkdir(exist_ok=False)
    frame=pd.DataFrame(rows);frame.to_csv(report/'learned_metrics.csv',index=False)
    summary=frame.groupby('case')[['mae','rmse']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(report/'learned_summary.csv')
    pd.DataFrame(fixed).to_csv(report/'fixed_metrics.csv',index=False)
    pd.DataFrame(scalar_rows).to_csv(report/'scalar_controls.csv',index=False)
    pd.DataFrame(proxy_rows).to_csv(report/'proxy_controls.csv',index=False)
    pd.DataFrame(per_window).to_csv(report/'per_window_metrics.csv',index=False)
    flat=[{k:v for k,v in a.items() if k!='phase_audit'} for a in audits]
    pd.DataFrame(flat).to_csv(report/'training_diagnostics.csv',index=False)
    (report/'audit.json').write_text(json.dumps({'status':'All paired initialization, shared-feature, update-budget, no-corruption/no-repair/no-test and scale-identity checks passed',
        'runs':audits,'scope':'Validation exploration, all cases reported; not full EEG TASK reproduction or clean-signal recovery'},indent=2),encoding='utf-8')
    plot(report,summary,fixed)
    print(summary.to_string(),flush=True)


def plot(out,summary,fixed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,metric in zip(axes,['mae','rmse']):
        names=summary.index.tolist()
        ax.bar(names,summary[metric+'_mean'],yerr=summary[metric+'_std'],capsize=4)
        for name,style in [('raw','--'),('last_day_persistence',':')]:
            value=next(r[metric] for r in fixed if r['method']==name)
            ax.axhline(value,color='black' if name=='raw' else 'tab:red',ls=style,label=name)
        ax.tick_params(axis='x',labelrotation=25)
        ax.set_ylabel('Validation '+metric.upper()+' (lower is better)')
        ax.legend(fontsize=8)
    fig.suptitle('Original single-variable data; all available daily-stride windows; 3 training seeds')
    fig.savefig(out/'alignment_results.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
