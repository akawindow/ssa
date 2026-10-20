"""Original observations, no corruption/repair: six matched neural runs plus static SSA diagnostics."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='runs/ssa_only_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'configs/ssa_only_protocol.json').read_text(encoding='utf-8'))
    base=json.loads((root/protocol['base_config']).read_text(encoding='utf-8'))
    assert base['repair_backend']=='none' and base['input_condition']=='reference'
    assert base['noise_std']==0 and base['outlier_rate']==0 and base['evaluation_splits']==['val']
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'configs').mkdir();manifest=[]
    for seed in protocol['training_seeds']:
        for mode in protocol['training_modes']:
            name=f'{mode}_seed{seed}';config=dict(base,seed=seed,training_mode=mode)
            path=out/'configs'/f'{name}.json';path.write_text(json.dumps(config,indent=2),encoding='utf-8')
            print(f'Running original observations: {name}',flush=True)
            with (out/f'{name}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-X','utf8','-m','ssa_task_srd','--config',str(path),'--out',str(out/name)],
                    cwd=root,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT)
            manifest.append({'training_seed':seed,'training_mode':mode,'directory':name,'exit_code':result.returncode})
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            if result.returncode:raise RuntimeError(f'{name} failed; inspect its log')
    print('Running fixed-rank and component-deletion diagnostics...',flush=True)
    subprocess.run([sys.executable,'-X','utf8','scripts/diagnose_ssa_only.py','--out',str(out/'static_diagnostics')],
        cwd=root,check=True)
    print(f'Completed six SSA-only training runs and static diagnostics: {out}',flush=True)


if __name__=='__main__':main()
