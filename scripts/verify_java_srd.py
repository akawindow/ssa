"""Run the unchanged author smoke command and verify our batch adapter against it."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from ssa_task_srd.java_srd import JavaSRD,sha256
from ssa_task_srd.repair import repair


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=False)
    java=JavaSRD(root);r=java.runtime;repo=Path(r['author_repository'])
    def execute(command,log,cwd=repo):
        result=subprocess.run([str(v) for v in command],cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        (out/log).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(f'Command failed; see {out/log}')
        return result.stdout
    stdout=execute([r['java_executable'],'-jar',r['author_jar'],'smoke'],'author_smoke.log')
    if 'Smoke test completed.' not in stdout:raise RuntimeError('Author smoke did not report success')
    canonical=repo/'results/smoke/srd_repaired.csv'
    shutil.copyfile(canonical,out/'author_smoke_repaired.csv')
    execute(java.command('smoke-fixture',repo/'data/real_clean/bank_1_7983.csv',out/'smoke_fixture.csv'),'smoke_fixture.log')
    fixture=pd.read_csv(out/'smoke_fixture.csv',float_precision='round_trip')
    original=pd.read_csv(canonical,float_precision='round_trip').value.to_numpy()
    np.testing.assert_allclose(fixture.repaired,original,atol=0,rtol=0)
    bridged,batch_audit=java.repair_batch(fixture.dirty.to_numpy()[None,:],288,6.0,5,out/'smoke_batch')
    np.testing.assert_allclose(bridged[0],original,atol=0,rtol=0)
    rows=[]
    def record(name,frame,period,k,files):
        before=float(np.sqrt(np.square(frame.dirty-frame.clean).mean()))
        after=float(np.sqrt(np.square(frame.repaired-frame.clean).mean()))
        rows.append({'case':name,'points':len(frame),'period':period,'k':k,'max_iterations':5,
            'dirty_rmse':before,'repaired_rmse':after,'relative_rmse_reduction':1-after/before if before else None,
            'changed_fraction':float(np.mean(frame.repaired.to_numpy()!=frame.dirty.to_numpy())),
            'source_sha256':json.dumps({str(p.relative_to(repo)):sha256(p) for p in files})})
    record('author_smoke_seed666',fixture,288,6.0,[repo/'data/real_clean/bank_1_7983.csv'])
    # Fixed representative cases selected by name, not by observed scores.
    for name,period in [('bank_1_7983.csv',288),('grid_1_10108.csv',1440),('telecom_1_12672.csv',288)]:
        clean=repo/'data/real_clean'/name;dirty=repo/'data/real_dirty'/name
        path=out/('pair_'+name)
        execute(java.command('real-pair',clean,dirty,period,9.0,5,path),name+'.log')
        frame=pd.read_csv(path,float_precision='round_trip')
        record(name,frame,period,9.0,[clean,dirty])
    python_repair=repair(fixture.dirty.to_numpy(),288,6.0,5)
    comparison={'legacy_python_vs_java_max_abs_difference':float(np.abs(python_repair-original).max()),
        'legacy_python_vs_java_rmse_difference':float(np.sqrt(np.square(python_repair-original).mean())),
        'legacy_python_is_not_relabelled_as_author_srd':True}
    results=pd.DataFrame(rows);results.to_csv(out/'repair_metrics.csv',index=False)
    report={'java_provenance':java.provenance(),'author_smoke_vs_fixture_max_abs_difference':0.0,
        'python_batch_adapter_vs_author_smoke_max_abs_difference':0.0,'smoke_batch':batch_audit,
        'comparison_with_legacy_prototype':comparison,
        'real_pair_normalization':'Matches author LoadData and Experiment.main_real: clean and dirty normalized separately',
        'scope':'Author code/adapter verification and three named paired cases, not full paper reproduction',
        'smoke_command_output':stdout,'test_forecast_data_evaluated':False}
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    certificate={key:r[key] for key in ['author_commit','author_jar_sha256','bridge_source_sha256','bridge_class_sha256']}
    certificate.update(max_abs_difference=0.0,verification_record=str(out/'verification.json'))
    (root/'data/java/bridge_verification.json').write_text(json.dumps(certificate,indent=2),encoding='utf-8')
    print(results.drop(columns='source_sha256').to_string(index=False));print(json.dumps(comparison,indent=2))
    print('Exact author-smoke / fixture / batch agreement passed.',flush=True)


if __name__=='__main__':main()
