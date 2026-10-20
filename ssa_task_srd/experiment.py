from pathlib import Path
import hashlib
import inspect
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from . import ssa
from .data import load_series, fit_scale, observed_input, make_windows
from .repair import repair, IMPLEMENTATION


def metrics(prediction, target, scale=1.0):
    err = (np.asarray(prediction)-np.asarray(target))*scale
    return {'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.square(err).mean()))}


def history_metrics(repaired, reference, outlier_mask):
    error = np.square(repaired-reference)
    def masked(m):
        return float(np.sqrt(error[m].mean())) if m.any() else None
    return {'history_rmse_z':float(np.sqrt(error.mean())),
        'outlier_rmse_z':masked(outlier_mask),'non_outlier_rmse_z':masked(~outlier_mask)}


def validate_config(config):
    keys=['period','history','horizon','stride','embedding','rank']
    if config.get('repair_backend','prototype')!='none':keys.append('srd_iterations')
    for key in keys:
        if not isinstance(config[key],int) or config[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    if config.get('repair_backend','prototype')!='none' and (config['period']%2 or config['history'] < 2*config['period']):
        raise ValueError('Repair prototype requires an even period and >=2 cycles')
    if not 2 <= config['embedding'] < config['history']:
        raise ValueError('Invalid embedding')
    if config['rank']>min(config['embedding'],config['history']-config['embedding']+1):
        raise ValueError('Invalid retained rank')
    if not 0 <= config['outlier_rate'] <= 1:
        raise ValueError('Invalid outlier rate')
    if config.get('evaluation_splits', ['val','test']) not in (['val'], ['val','test']):
        raise ValueError('evaluation_splits must be [val] or [val, test]')
    if config.get('selector_mode','raw_conv') not in ['raw_conv','proxy_features','random_features']:
        raise ValueError('Invalid selector mode')
    if config.get('repair_backend','prototype') not in ['prototype','java','none']:
        raise ValueError('Invalid repair backend; select prototype, java or none explicitly')
    if config.get('training_mode','legacy_frozen') not in ['legacy_frozen','paired_frozen','collaborative']:
        raise ValueError('Invalid training mode')
    if config.get('training_mode','legacy_frozen')!='legacy_frozen' and config.get('selector_mode')!='proxy_features':
        raise ValueError('Paired training comparison requires shared proxy_features')


def cached_components(root, windows, config):
    hasher=hashlib.sha256()
    hasher.update(inspect.getsource(ssa.decompose).encode())
    hasher.update(json.dumps([config['embedding'],config['rank'],np.__version__]).encode())
    for split,data in windows.items():
        hasher.update(data['x'].tobytes());hasher.update(data['origins'].tobytes())
    key=hasher.hexdigest()[:20]
    path=Path(root)/'data'/'cache'/f'ssa_{key}.npz'
    start=time.perf_counter()
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            result={split:z[split] for split in windows}
        return result,{'hit':True,'seconds':time.perf_counter()-start,'key':key,'bytes':path.stat().st_size}
    result={}
    for split,data in windows.items():
        print(f'SSA decomposition: {split}, {len(data["x"])} histories',flush=True)
        result[split]=np.array([ssa.decompose(x,config['embedding'],config['rank']) for x in data['x']])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**result)
    return result,{'hit':False,'seconds':time.perf_counter()-start,'key':key,'bytes':path.stat().st_size}


def make_plots(out,times,values,observed,origin,windows,variants,rows,training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
    stop=min(len(values),14*144)
    axes[0].plot(times[:stop],values[:stop],lw=.7)
    axes[0].set_title('Original observations (first 14 days; not assumed perfectly clean)')
    axes[0].set_ylabel('Power consumption')
    start=origin-windows['train']['x'].shape[1]
    axes[1].plot(times[start:origin],values[start:origin],lw=1,label='Reference observation')
    original_input=np.allclose(values,observed,rtol=0,atol=1e-10)
    axes[1].plot(times[start:origin],observed[start:origin],lw=.6,alpha=.7,
        label='Model input (no added corruption)' if original_input else 'Contaminated history')
    axes[1].set_title('Training input example; no added corruption' if original_input else
        'Training input example; perturbations applied only to model inputs')
    axes[1].legend();fig.savefig(out/'data_preview.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    display_split = 'test' if 'test' in windows else 'val'
    ax.plot(windows[display_split]['reference'][0],color='black',lw=1,label='Reference')
    for name,split_data in variants.items():
        ax.plot(split_data[display_split][0],lw=.8,alpha=.75,label=name)
    ax.set_title(f'Example processed history ({display_split}; exploratory)')
    ax.legend(ncol=3,fontsize=8);fig.savefig(out/'processing_preview.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    selected=[r for r in rows if r['split']==display_split]
    ax.bar([r['method'] for r in selected],[r['forecast_rmse_original'] for r in selected])
    ax.tick_params(axis='x',labelrotation=20)
    ax.set_ylabel('Forecast RMSE (original units)')
    ax.set_title(f'{display_split}: single-seed pilot; not evidence of general superiority')
    fig.savefig(out/'forecast_comparison.png',dpi=140);plt.close(fig)
    if training:
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        for ax,name in zip(axes,['proxy_loss','selector_loss']):
            ax.plot(training[name]['loss'],label='train');ax.plot(training[name]['val_loss'],label='validation')
            ax.set_title(name);ax.legend()
        fig.savefig(out/'training_curves.png',dpi=140);plt.close(fig)


def run(config, root, out, task_driven=True):
    validate_config(config)
    root,out=Path(root),Path(out)
    if out.exists():
        raise FileExistsError(f'Output directory exists; use a new run name: {out}')
    out.mkdir(parents=True)
    started=time.perf_counter()
    reference,times,audit=load_series(config,root)
    train_end=int(.7*len(reference))
    mean,std=fit_scale(reference,train_end)
    reference_z=(reference-mean)/std
    observed_z,mask=observed_input(reference_z,config)
    windows,boundaries=make_windows(reference_z,observed_z,mask,config)
    audit.update({'split_boundaries':boundaries,'scale_mean':mean,'scale_std':std,'scaler_fit_range':[0,train_end],
        'column':config.get('column','synthetic'),'period_points':config['period'],
        'training_lag_period_correlation':float(np.corrcoef(reference_z[:train_end-config['period']],reference_z[config['period']:train_end])[0,1]),
        'window_counts':{k:len(v['x']) for k,v in windows.items()},
        'sampled_origins':{k:v['origins'].tolist() for k,v in windows.items()},
        'target_ranges':{k:[[int(o),int(o+config['horizon'])] for o in v['origins']] for k,v in windows.items()},
        'scope':'Engineering subset sampled evenly over chronological splits; NOT a full benchmark'})
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    components,cache=cached_components(root,windows,config)
    timings={'ssa':cache}
    backend=config.get('repair_backend','prototype')
    repair_enabled=backend!='none'
    java_srd=None
    if backend=='java':
        from .java_srd import JavaSRD
        java_srd=JavaSRD(root)
        if not java_srd.provenance()['adapter_verified_against_author_smoke']:
            raise RuntimeError('Java adapter is not verified against author smoke; run scripts/verify_java_srd.py first')
    repair_name='srd_java' if java_srd else 'srd_prototype'
    variants={'raw':{k:v['x'] for k,v in windows.items()},
        'ssa':{k:np.array([ssa.ordinary(c) for c in v]) for k,v in components.items()}}
    def clean_variant(name,source):
        start=time.perf_counter()
        batches={}
        if java_srd:
            variants[name]={}
            for split,a in source.items():
                repaired,audit=java_srd.repair_batch(a,config['period'],config['srd_k'],config['srd_iterations'],
                    artifact_dir=out/'repair_io'/name/split)
                variants[name][split]=repaired;batches[split]=audit
        else:
            variants[name]={k:np.array([repair(x,config['period'],config['srd_k'],config['srd_iterations']) for x in a]) for k,a in source.items()}
        timings[name]={'seconds':time.perf_counter()-start,'java_batches':batches}
        np.savez_compressed(out/f'histories_{name}.npz',**variants[name])
    if repair_enabled:
        clean_variant(repair_name,variants['raw'])
        clean_variant('ssa_'+repair_name,variants['ssa'])
    training=None
    if task_driven:
        from .learning import learn
        start=time.perf_counter()
        rebuilt,weights,training,proxy,selector=learn(config,windows,components)
        timings['learning']={'seconds':time.perf_counter()-start}
        variants['task_ssa']=rebuilt
        if repair_enabled:clean_variant('task_ssa_'+repair_name,rebuilt)
        np.savez_compressed(out/'histories_task_ssa.npz',**rebuilt)
        np.savez_compressed(out/'component_weights.npz',**weights)
        proxy.save_weights(str(out/'proxy.weights.h5'))
        selector.save_weights(str(out/'selector.weights.h5'))
        (out/'training.json').write_text(json.dumps(training,indent=2),encoding='utf-8')
    rows=[]
    for name,inputs in variants.items():
        print(f'Evaluating independent ridge forecaster: {name}',flush=True)
        model=Ridge(alpha=config['ridge_alpha'],solver=config.get('ridge_solver','lsqr'))
        readout_inputs=({k:np.asarray(v,dtype=np.float64) for k,v in inputs.items()}
            if config.get('ridge_solver')=='cholesky' else inputs)
        model.fit(readout_inputs['train'],windows['train']['y'])
        for split in config.get('evaluation_splits',['val','test']):
            prediction=model.predict(readout_inputs[split])
            score=metrics(prediction,windows[split]['y'],std)
            rows.append({'method':name,'split':split,'forecast_mae_original':score['mae'],'forecast_rmse_original':score['rmse'],
                **history_metrics(inputs[split],windows[split]['reference'],windows[split]['outlier_mask'])})
            np.savez_compressed(out/f'predictions_{name}_{split}.npz',prediction_z=prediction,target_z=windows[split]['y'],origins=windows[split]['origins'])
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    versions={}
    for package in ['numpy','scipy','pandas','scikit-learn','matplotlib','tensorflow']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    try: git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError): git_commit=None
    meta={'created_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'platform':platform.platform(),'versions':versions,'git_commit':git_commit,'seed':config['seed'],
        'repair_implementation':('disabled_no_repair' if not repair_enabled else
            ('original_author_java_'+java_srd.runtime['author_commit']) if java_srd else IMPLEMENTATION),
        'repair_backend':backend,'original_author_java_called':java_srd is not None,
        'repair_enabled':repair_enabled,'input_condition':config.get('input_condition'),
        'artificial_corruption_applied':bool(config.get('noise_std',0)>0 or config.get('outlier_rate',0)>0),
        'java_srd_provenance':java_srd.provenance() if java_srd else None,
        'original_java_parity_verified':bool(java_srd and java_srd.provenance()['adapter_verified_against_author_smoke']),
        'training_mode':config.get('training_mode','legacy_frozen'),
        'pipeline_training':(('Alternating selector/proxy training' if config.get('training_mode')=='collaborative'
            else 'Frozen proxy supervises selector')+(' BEFORE repair; no backprop through repair' if repair_enabled else
            '; SSA only, repair disabled')),
        'forecaster':'Independent Ridge, refit per method with the same alpha',
        'ridge_solver':config.get('ridge_solver','lsqr'),
        'evaluation_splits':config.get('evaluation_splits',['val','test']),
        'test_windows_constructed':'test' in windows,
        'selector_mode':config.get('selector_mode','raw_conv'),
        'corruption_seed':config.get('corruption_seed',config['seed']),
        'limitations':['Single seed, small evenly sampled subset','Reference observations are not guaranteed clean',
            ('SSA-only; no repair algorithm used' if not repair_enabled else
             'Pinned author Java implementation; not a complete paper replication' if java_srd else
             'SRD-style Python prototype; not original-author SRD equivalence'),'No claim of novelty or effectiveness',
            'Each run has one input condition; no noise-source sweep or multiple final predictors'],
        'timings':timings,'elapsed_seconds_before_plots':time.perf_counter()-started}
    make_plots(out,times,reference,observed_z*std+mean,int(windows['train']['origins'][0]),windows,variants,rows,training)
    meta['elapsed_seconds_total']=time.perf_counter()-started
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    np.savez_compressed(out/'perturbation.npz',outlier_mask=mask,observed_z=observed_z,reference_z=reference_z)
    display_split = 'test' if 'test' in windows else 'val'
    print(pd.DataFrame(rows).query('split == @display_split').to_string(index=False),flush=True)
    print(f'Completed: {out}; {meta["elapsed_seconds_total"]:.2f}s',flush=True)
    return rows,meta
