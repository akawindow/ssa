"""Primary task-model experiment with separate fit, selection and assessment roles."""
import os
for key in ['TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
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
from ssa_task_srd.data import load_series,fit_scale,observed_input,make_windows
from ssa_task_srd.experiment import cached_components,metrics
from ssa_task_srd.learning import tensorflow,build_proxy,weight_digest,weight_summary
from ssa_task_srd.mlp_primary import train_branch,predict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/mlp_primary_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    protocol=json.loads((root/'configs/mlp_primary_protocol.json').read_text(encoding='utf-8'))
    config=json.loads((root/protocol['base_config']).read_text(encoding='utf-8'))
    config.update(protocol['dataset_overrides']);config.update(protocol['training'])
    config.update(rank=144,training_mode='collaborative')
    assert config['input_condition']=='reference' and config['noise_std']==config['outlier_rate']==0
    assert config['repair_backend']=='none' and config['evaluation_splits']==['val']
    values,times,audit=load_series(config,root)
    provisional,split_boundaries=make_windows(values,values,np.zeros(len(values),dtype=bool),config)
    nfit=int(len(provisional['train']['x'])*protocol['fit_fraction_of_existing_training_windows'])
    cutoff=int(provisional['train']['origins'][nfit])
    mean,std=fit_scale(values,cutoff)
    reference=(values-mean)/std;observed,mask=observed_input(reference,config)
    np.testing.assert_array_equal(reference,observed);assert not mask.any()
    original,_=make_windows(reference,observed,mask,config)
    windows={name:{k:v[section] for k,v in original['train'].items()}
        for name,section in [('fit',slice(None,nfit)),('selection',slice(nfit,None))]}
    windows['assessment']=original['val']
    assert windows['fit']['origins'][-1]+config['horizon']<=windows['selection']['origins'][0]
    assert windows['selection']['origins'][-1]+config['horizon']<=windows['assessment']['origins'][0]
    components,cache=cached_components(root,windows,config)
    for name,w in windows.items():np.testing.assert_allclose(components[name].sum(axis=1),w['x'],atol=1e-10,rtol=0)
    raw={k:w['x'] for k,w in windows.items()}
    prepared={'raw_continue':raw,'half_continue':{k:x*.5 for k,x in raw.items()},
              'ssa7_continue':{k:c[:,:7,:].sum(axis=1) for k,c in components.items()},
              'global_gate':raw,'task_full':components}
    audit.update({'original_split_boundaries':split_boundaries,'window_counts':{k:len(w['x']) for k,w in windows.items()},
        'origins':{k:w['origins'].tolist() for k,w in windows.items()},'scale_mean':mean,'scale_std':std,
        'standardization_fit_range':[0,cutoff],'last_scaler_fit_time':str(times[cutoff-1]),
        'target_ranges':{k:[[int(o),int(o+config['horizon'])] for o in w['origins']] for k,w in windows.items()},
        'artificial_corruption':False,'repair':False,'test_windows_constructed':False,'independent_ridge_evaluated':False,
        'assessment_previously_explored':True,'assessment_used_for_new_checkpoints':False,'cache':cache})
    for name,value in [('protocol',protocol),('config',config),('data_audit',audit)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print('Data roles:',audit['window_counts'],'; scaler fits before index',cutoff,flush=True)
    rows=[];selection_rows=[];per_window=[];training_audits=[];manifest=[]
    def record(branch,seed,prediction,selection_prediction):
        score=metrics(prediction,windows['assessment']['y'],std)
        select_score=metrics(selection_prediction,windows['selection']['y'],std)
        rows.append({'branch':branch,'seed':seed,'mae':score['mae'],'rmse':score['rmse']})
        selection_rows.append({'branch':branch,'seed':seed,'mae':select_score['mae'],'rmse':select_score['rmse']})
        for origin,p,y in zip(windows['assessment']['origins'],prediction,windows['assessment']['y']):
            m=metrics(p,y,std);per_window.append({'branch':branch,'seed':seed,'origin':int(origin),'mae':m['mae'],'rmse':m['rmse']})
        print(f'{branch}, seed={seed}: assessment MAE={score["mae"]:.2f}, RMSE={score["rmse"]:.2f}',flush=True)
    for seed in protocol['seeds']:
        tf=tensorflow(seed);local=dict(config,seed=seed)
        base=build_proxy(tf,config['history'],config['horizon'])
        base.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),loss='mse')
        folder=out/f'pretraining_seed{seed}';folder.mkdir()
        pretrain_checkpoint=folder/'best_selection.weights.h5'
        pretraining=base.fit(raw['fit'],windows['fit']['y'],validation_data=(raw['selection'],windows['selection']['y']),
            epochs=config['proxy_epochs'],batch_size=config['batch_size'],verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=config['patience'],restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(str(pretrain_checkpoint),monitor='val_loss',save_best_only=True,save_weights_only=True)])
        # Explicitly restore even when the maximum epoch is reached without an
        # early-stop event; never rely on callback-version restoration details.
        base.load_weights(str(pretrain_checkpoint))
        pre_selection=base(raw['selection'],training=False).numpy()
        np.testing.assert_allclose(np.square(pre_selection-windows['selection']['y']).mean(),
            min(pretraining.history['val_loss']),rtol=1e-5,atol=1e-7)
        initial=[a.copy() for a in base.get_weights()];initial_digest=weight_digest(initial)
        base.save_weights(str(folder/'proxy.weights.h5'))
        (folder/'training.json').write_text(json.dumps({'seed':seed,'pretrained_proxy_sha256':initial_digest,
            'history':pretraining.history,'selection_only':True,'best_weights_explicitly_restored':True,
            'best_pretraining_epoch':int(np.argmin(pretraining.history['val_loss']))+1},indent=2),encoding='utf-8')
        pre_assessment=base(raw['assessment'],training=False).numpy()
        record('raw_pretrained',seed,pre_assessment,pre_selection)
        np.savez_compressed(folder/'predictions.npz',assessment=pre_assessment,selection=pre_selection,
            target_z=windows['assessment']['y'],origins=windows['assessment']['origins'])
        del base
        for branch in protocol['branches']:
            tick=time.perf_counter();tf.keras.backend.clear_session()
            proxy=build_proxy(tf,config['history'],config['horizon']);proxy.set_weights(initial)
            inputs=prepared[branch]
            gate,training=train_branch(tf,proxy,branch,inputs['fit'],windows['fit']['y'],
                inputs['selection'],windows['selection']['y'],local)
            assert training['initial_proxy_sha256']==initial_digest
            # Assessment targets are first used here, after checkpoint restoration.
            assessment=predict(proxy,gate,inputs['assessment'])
            selection=predict(proxy,gate,inputs['selection'])
            record(branch,seed,assessment,selection)
            folder=out/f'{branch}_seed{seed}';folder.mkdir()
            proxy.save_weights(str(folder/'proxy.weights.h5'))
            if branch=='task_full':
                gate.head.save_weights(str(folder/'head.weights.h5'))
                weights={k:gate.reconstruct(x)[1].numpy()[...,0] for k,x in inputs.items()}
                np.savez_compressed(folder/'component_weights.npz',**weights)
                training['retention_summary']={k:weight_summary(w) for k,w in weights.items()}
            training['elapsed_seconds']=time.perf_counter()-tick
            (folder/'training.json').write_text(json.dumps(training,indent=2),encoding='utf-8')
            np.savez_compressed(folder/'predictions.npz',assessment=assessment,selection=selection,
                target_z=windows['assessment']['y'],origins=windows['assessment']['origins'])
            training_audits.append({k:v for k,v in training.items() if k not in ['curve','retention_summary']})
            manifest.append({'branch':branch,'seed':seed,'directory':folder.name,'completed':True})
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            del gate,proxy
    naive=raw['assessment'][:,-config['horizon']:];naive_selection=raw['selection'][:,-config['horizon']:]
    record('last_day_persistence',None,naive,naive_selection)
    for seed in protocol['seeds']:
        matched=[a for a in training_audits if a['seed']==seed]
        assert len(matched)==len(protocol['branches'])
        for field in ['initial_proxy_sha256','batch_order_sha256','proxy_optimizer_steps','proxy_parameters']:
            assert len({a[field] for a in matched})==1,field
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(selection_rows).to_csv(out/'selection_metrics.csv',index=False)
    pd.DataFrame(per_window).to_csv(out/'per_window_metrics.csv',index=False)
    neural=frame[frame.branch!='last_day_persistence']
    summary=neural.groupby('branch')[['mae','rmse']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(out/'summary.csv')
    wide=neural.pivot(index='seed',columns='branch',values='rmse');paired=[]
    for control in ['raw_continue','half_continue','ssa7_continue','global_gate']:
        for seed,delta in (wide.task_full-wide[control]).items():
            paired.append({'control':control,'seed':seed,'task_full_minus_control_rmse':float(delta)})
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    (out/'training_audit.json').write_text(json.dumps({'status':'Common initialization and proxy budget, phase isolation, coupled restoration, selection-only checkpoints passed',
        'runs':training_audits,'scope':'Primary same-architecture MLP comparison; historical assessment remains exploratory'},indent=2),encoding='utf-8')
    plot(out,summary,float(frame.loc[frame.branch=='last_day_persistence','rmse'].iloc[0]))
    (out/'metadata.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'elapsed_seconds':time.perf_counter()-started,'seeds':protocol['seeds'],'training_branches':len(manifest),
        'primary_evaluation':'Same paired trained MLP on own branch inputs','independent_ridge_evaluated':False,
        'assessment_used_for_new_checkpoints':False,'virgin_holdout':False},indent=2),encoding='utf-8')
    print(summary.to_string());print('Primary MLP experiment completed with matched proxy budgets.')


def plot(out,summary,naive_rmse):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['raw_pretrained','raw_continue','half_continue','ssa7_continue','global_gate','task_full']
    s=summary.loc[order];fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    ax.bar(order,s.rmse_mean,yerr=s.rmse_std,capsize=4,color=['grey','tab:blue','tab:orange','tab:green','tab:purple','tab:red'])
    ax.axhline(naive_rmse,color='black',ls='--',label='Yesterday same time')
    ax.tick_params(axis='x',labelrotation=25);ax.set_ylabel('Assessment RMSE (lower is better)')
    ax.set_title('Primary MLP comparison; checkpoints selected on a separate training-period block')
    ax.legend();fig.savefig(out/'mlp_primary_results.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
