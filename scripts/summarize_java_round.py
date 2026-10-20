"""Export versionable original-Java evidence without copying author source/data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ssa_task_srd.data import make_windows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--verification',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    source=root/args.run;verification=root/args.verification;out=root/args.out
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if len(manifest)!=6 or any(m['exit_code'] for m in manifest):raise RuntimeError('Java matrix incomplete')
    rows=[];audits=[];changes=[];comparisons=[]
    for entry in manifest:
        folder=source/entry['directory'];config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        training=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        table=pd.read_csv(folder/'metrics.csv')
        assert set(table.split)=={'val'} and not meta['test_windows_constructed']
        assert meta['original_author_java_called'] and meta['original_java_parity_verified']
        with np.load(folder/'perturbation.npz') as z:
            windows,_=make_windows(z['reference_z'],z['observed_z'],z['outlier_mask'],config)
        for record in table.to_dict('records'):rows.append(dict(record,condition=entry['condition'],k=entry['k']))
        for name in ['srd_java','ssa_srd_java','task_ssa_srd_java']:
            with np.load(folder/f'histories_{name}.npz') as z:
                if name=='srd_java':
                    for split,window in windows.items():
                        delta=z[split]-window['x']
                        changes.append({'condition':entry['condition'],'k':entry['k'],'split':split,
                            'changed_fraction':float(np.mean(np.abs(delta)>1e-12)),
                            'max_abs_change_z':float(np.abs(delta).max())})
        paired_old=root/'runs/round02_initial'/f'{entry["condition"]}_proxy_features_seed42'
        # Optional historical comparison; Java baseline does not depend on legacy run availability.
        if paired_old.exists():
            previous=pd.read_csv(paired_old/'metrics.csv')
            old_training=json.loads((paired_old/'training.json').read_text(encoding='utf-8'))
            assert old_training['pretrained_proxy_sha256']==training['pretrained_proxy_sha256']
            for name in ['raw','ssa','task_ssa']:
                np.testing.assert_allclose(previous.loc[previous.method==name,'forecast_rmse_original'],
                    table.loc[table.method==name,'forecast_rmse_original'],rtol=0,atol=1e-8)
            if entry['k']==3:
                for old,new in [('srd_prototype','srd_java'),('ssa_srd_prototype','ssa_srd_java'),('task_ssa_srd_prototype','task_ssa_srd_java')]:
                    before=float(previous.loc[previous.method==old,'forecast_rmse_original'].iloc[0])
                    after=float(table.loc[table.method==new,'forecast_rmse_original'].iloc[0])
                    comparisons.append({'condition':entry['condition'],'k':3,'method':new,
                        'legacy_python_rmse':before,'author_java_rmse':after,'java_minus_python':after-before})
        audits.append({'condition':entry['condition'],'k':entry['k'],'code_commit':meta['git_commit'],
            'proxy_sha256':training['pretrained_proxy_sha256'],'proxy_frozen':training['proxy_weights_unchanged'],
            'java':meta['java_srd_provenance'],'java_batches':{name:meta['timings'][name]['java_batches']
                for name in ['srd_java','ssa_srd_java','task_ssa_srd_java']},
            'elapsed_seconds':meta['elapsed_seconds_total']})
    assert len({a['proxy_sha256'] for a in audits})==1
    frame=pd.DataFrame(rows);frame.to_csv(out/'forecast_metrics.csv',index=False)
    pd.DataFrame(changes).to_csv(out/'normal_input_changes.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'legacy_comparison_k3.csv',index=False)
    shutil.copyfile(verification/'repair_metrics.csv',out/'author_repair_metrics.csv')
    evidence=json.loads((verification/'verification.json').read_text(encoding='utf-8'))
    (out/'author_verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (out/'integration_audit.json').write_text(json.dumps({'runs':6,'test_evaluation':False,'runs_metadata':audits,
        'scope':'One training seed, three declared thresholds, two input conditions; not a complete SRD paper reproduction'},indent=2),encoding='utf-8')
    runtime=json.loads((root/'data/java/runtime.json').read_text(encoding='utf-8'))
    public_keys=['author_commit','author_url','java_version','maven_version','author_source_hashes','author_jar_sha256',
        'bridge_source_sha256','bridge_class_sha256','author_tracked_files_unchanged','global_environment_modified']
    (out/'toolchain.json').write_text(json.dumps({key:runtime[key] for key in public_keys},indent=2),encoding='utf-8')
    plot(out,frame)
    print(frame[frame.method.isin(['raw','srd_java','task_ssa_srd_java'])][['condition','k','method','forecast_rmse_original']].to_string(index=False))
    print(pd.DataFrame(changes).to_string(index=False))
    print('Audits passed; no author source, raw data, or model weights exported.')


def plot(out,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,condition in zip(axes,['reference','mixed']):
        subset=frame[frame.condition==condition]
        ax.axhline(subset[subset.method=='raw'].forecast_rmse_original.iloc[0],color='black',ls='--',label='Raw')
        for name,label in [('srd_java','Author SRD'),('task_ssa_srd_java','Task SSA + author SRD')]:
            scores=subset[subset.method==name].sort_values('k')
            ax.plot(scores.k,scores.forecast_rmse_original,marker='o',label=label)
        ax.set_xticks([3,6,9]);ax.set_xlabel('Residual threshold k');ax.set_ylabel('Validation forecast RMSE')
        ax.set_title('No added corruption' if condition=='reference' else 'Fixed mixed corruption');ax.legend(fontsize=8)
    fig.suptitle('Author Java backend; one training seed; all declared thresholds shown')
    fig.savefig(out/'java_validation.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
