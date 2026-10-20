"""Post-hoc validation diagnostic: reconstruct matched heads before/after training.

No new neural training or test evaluation; all outputs are new, never overwritten.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']:
    os.environ.setdefault(name,'1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from ssa_task_srd.data import make_windows
from ssa_task_srd.experiment import cached_components,metrics
from ssa_task_srd.learning import tensorflow,build_proxy,build_feature_selector,weight_digest
from ssa_task_srd.repair import repair


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/args.run;out=root/args.out
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    rows=[];checks=[]
    for entry in manifest:
        if entry['mode']=='raw_conv':continue
        folder=source/entry['directory']
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        training=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        audit=json.loads((folder/'data_audit.json').read_text(encoding='utf-8'))
        assert config['evaluation_splits']==['val']
        with np.load(folder/'perturbation.npz') as z:
            windows,_=make_windows(z['reference_z'],z['observed_z'],z['outlier_mask'],config)
        components,_=cached_components(root,windows,config)
        tf=tensorflow(config['seed'])
        proxy=build_proxy(tf,config['history'],config['horizon'])
        random_weights=[w.copy() for w in proxy.layers[0].get_weights()]
        proxy.load_weights(str(folder/'proxy.weights.h5'));proxy.trainable=False
        assert weight_digest(proxy.get_weights())==training['pretrained_proxy_sha256']
        selector=build_feature_selector(tf,proxy,config['history'],entry['mode'],random_weights)
        assert weight_digest([v.numpy() for v in selector.trainable_variables])==training['initial_trainable_selector_sha256']
        def reconstruct():
            rebuilt={};weight_arrays={}
            for split,c in components.items():
                shape=c.shape
                w=selector(tf.reshape(tf.cast(c,tf.float32),[-1,config['history'],1]),training=False)
                w=tf.reshape(w,[shape[0],shape[1],1])
                rebuilt[split]=tf.reduce_sum(tf.cast(c,tf.float32)*w,axis=1).numpy()
                weight_arrays[split]=w.numpy()[...,0]
            return rebuilt,weight_arrays
        initial,initial_weights=reconstruct()
        np.testing.assert_allclose(initial_weights['val'].mean(),training['initial_weight_summary']['val']['mean'],atol=1e-7,rtol=0)
        selector.load_weights(str(folder/'selector.weights.h5'))
        final,final_weights=reconstruct()
        with np.load(folder/'component_weights.npz') as z:
            np.testing.assert_allclose(final_weights['val'],z['val'],atol=1e-7,rtol=0)
        max_prediction_difference=0.0
        for state,source_inputs in [('initial',initial),('trained',final)]:
            processed={split:np.array([repair(x,config['period'],config['srd_k'],config['srd_iterations']) for x in a])
                for split,a in source_inputs.items()}
            for method,inputs in [('task_ssa',source_inputs),('task_ssa_srd_prototype',processed)]:
                model=Ridge(alpha=config['ridge_alpha'],solver='lsqr').fit(inputs['train'],windows['train']['y'])
                prediction=model.predict(inputs['val'])
                if state=='trained':
                    with np.load(folder/f'predictions_{method}_val.npz') as z:
                        np.testing.assert_allclose(prediction,z['prediction_z'],rtol=1e-5,atol=1e-6)
                        max_prediction_difference=max(max_prediction_difference,float(np.max(np.abs(prediction-z['prediction_z']))))
                score=metrics(prediction,windows['val']['y'],audit['scale_std'])
                rows.append({'condition':entry['condition'],'mode':entry['mode'],'training_seed':entry['training_seed'],
                    'state':state,'method':method,'forecast_rmse_original':score['rmse'],'forecast_mae_original':score['mae']})
        checks.append({'directory':entry['directory'],'initial_head_and_weight_summary_match':True,
            'trained_weights_match':True,'max_prediction_difference_z':max_prediction_difference})
        tf.keras.backend.clear_session()
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'initial_vs_trained.csv',index=False)
    summary=frame.groupby(['condition','mode','method','state']).forecast_rmse_original.agg(['mean','std']).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    pivot=frame.pivot(index=['condition','mode','method','training_seed'],columns='state',values='forecast_rmse_original')
    pivot['trained_minus_initial_rmse']=pivot['trained']-pivot['initial']
    paired=pivot.reset_index().groupby(['condition','mode','method']).trained_minus_initial_rmse.agg(['mean','std']).reset_index()
    paired.to_csv(out/'paired_changes.csv',index=False)
    (out/'audit.json').write_text(json.dumps({'status':'Post-hoc diagnostic, not part of the locked primary matrix',
        'test_evaluated':False,'new_neural_training':False,'checks':checks},indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(paired.to_string(index=False))


if __name__=='__main__':main()
