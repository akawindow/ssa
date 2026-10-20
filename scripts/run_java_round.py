"""Six predeclared validation runs using the author's actual Java implementation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/java_round_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'configs/java_protocol.json').read_text(encoding='utf-8'))
    base=json.loads((root/protocol['base_config']).read_text(encoding='utf-8'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'configs').mkdir();manifest=[]
    for condition in protocol['conditions']:
        for k in protocol['thresholds']:
            name=f'{condition}_java_k{k:g}'
            config=dict(base,seed=protocol['training_seed'],corruption_seed=protocol['corruption_seed'],
                selector_mode=protocol['selector_mode'],repair_backend='java',srd_k=k,
                evaluation_splits=['val'],input_condition=condition)
            if condition=='reference':config.update(noise_std=0.0,outlier_rate=0.0)
            path=out/'configs'/f'{name}.json';path.write_text(json.dumps(config,indent=2),encoding='utf-8')
            print(f'Running {name}',flush=True)
            with (out/f'{name}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-X','utf8','-m','ssa_task_srd','--config',str(path),'--out',str(out/name)],
                    cwd=root,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT)
            manifest.append({'condition':condition,'k':k,'directory':name,'exit_code':result.returncode})
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            if result.returncode:raise RuntimeError(f'Run failed: {name}; inspect its log')
    print(f'Completed {len(manifest)} Java runs: {out}',flush=True)


if __name__=='__main__':main()
