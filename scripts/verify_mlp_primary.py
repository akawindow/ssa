"""Reload primary MLP checkpoints and verify inference without any training."""
import os
for key in ['TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ssa_task_srd.data import load_series,fit_scale
from ssa_task_srd.experiment import cached_components,metrics
from ssa_task_srd.learning import tensorflow,build_proxy,weight_digest
from ssa_task_srd.collaborative import build_alternating_task


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];run=root/args.run;out=root/args.out
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((run/'config.json').read_text());audit=json.loads((run/'data_audit.json').read_text())
    protocol=json.loads((run/'protocol.json').read_text())
    values,_,_=load_series(cfg,root)
    mean,std=fit_scale(values,audit['standardization_fit_range'][1])
    assert mean==audit['scale_mean'] and std==audit['scale_std']
    z=(values-mean)/std
    windows={k:{'x':np.array([z[o-cfg['history']:o] for o in origins]),'origins':np.array(origins)}
             for k,origins in audit['origins'].items()}
    components,_=cached_components(root,windows,cfg)
    target=np.array([z[o:o+cfg['horizon']] for o in audit['origins']['assessment']])
    tf=tensorflow(0);checks=[];shape_rows=[]
    scores=pd.read_csv(run/'metrics.csv').set_index(['branch','seed'])
    for seed in protocol['seeds']:
        for branch in ['raw_pretrained']+protocol['branches']:
            tf.keras.backend.clear_session()
            folder=run/(f'pretraining_seed{seed}' if branch=='raw_pretrained' else f'{branch}_seed{seed}')
            proxy=build_proxy(tf,cfg['history'],cfg['horizon']);proxy.load_weights(str(folder/'proxy.weights.h5'))
            x=windows['assessment']['x']
            training=json.loads((folder/'training.json').read_text())
            if branch=='half_continue':rebuilt=x*.5
            elif branch=='ssa7_continue':rebuilt=components['assessment'][:,:7,:].sum(axis=1)
            elif branch=='global_gate':rebuilt=tf.cast(x,tf.float32)*tf.constant(training['selected_global_scale'],tf.float32)
            elif branch=='task_full':
                gate=build_alternating_task(tf,proxy,cfg['history'],dict(cfg,n_train=len(windows['fit']['x'])))
                gate.head.load_weights(str(folder/'head.weights.h5'))
                rebuilt,weights=gate.reconstruct(components['assessment'])
                with np.load(folder/'component_weights.npz') as saved:
                    np.testing.assert_array_equal(weights.numpy()[...,0],saved['assessment'])
                fit_rebuilt=gate.reconstruct(components['fit'])[0].numpy().astype(np.float64)
                fit_raw=windows['fit']['x']
                scalar=float(np.sum(fit_raw*fit_rebuilt)/np.sum(fit_raw*fit_raw))
                shape_delta=rebuilt.numpy().astype(np.float64)-scalar*x
                shape_rows.append({'seed':seed,'fit_least_squares_scalar':scalar,
                    'assessment_shape_difference_rms_z':float(np.sqrt(np.square(shape_delta).mean())),
                    'assessment_shape_difference_over_input_rms':float(np.sqrt(np.square(shape_delta).mean()/np.square(x).mean())),
                    'assessment_retention_mean':float(weights.numpy().mean())})
            else:rebuilt=x
            prediction=proxy(rebuilt,training=False).numpy()
            with np.load(folder/'predictions.npz') as saved:
                delta=float(np.max(np.abs(prediction-saved['assessment']))*std)
                np.testing.assert_array_equal(saved['target_z'],target)
            assert delta<1e-5,(seed,branch,delta)
            score=metrics(prediction,target,std)
            np.testing.assert_allclose(score['rmse'],scores.loc[(branch,seed),'rmse'],atol=1e-6,rtol=0)
            expected=training['pretrained_proxy_sha256'] if branch=='raw_pretrained' else training['final_proxy_sha256']
            assert weight_digest(proxy.get_weights())==expected
            checks.append({'branch':branch,'seed':seed,'max_prediction_difference_original':delta,
                'rmse':score['rmse'],'proxy_sha256_matches':True,
                'checkpoint_sha256':hashlib.sha256((folder/'proxy.weights.h5').read_bytes()).hexdigest()})
    pd.DataFrame(checks).to_csv(out/'checkpoint_reload_checks.csv',index=False)
    pd.DataFrame(shape_rows).to_csv(out/'task_shape_diagnostics.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'source_run':str(run.relative_to(root)),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'models_reloaded':len(checks),'neural_training_performed':False,'assessment_not_used_for_selection':True,
        'shape_diagnostic':'Post-run description only; training histories determine least-squares scalar; not denoising accuracy or a new trained method'},indent=2),encoding='utf-8')
    print('All',len(checks),'saved MLP checkpoints and assessment predictions verified.')
    print(pd.DataFrame(shape_rows).to_string(index=False))


if __name__=='__main__':main()
