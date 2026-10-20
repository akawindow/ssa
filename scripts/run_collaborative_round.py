"""Run the locked alternating-vs-frozen comparison; never reuse unmatched old controls."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/collaborative_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'configs/collaborative_protocol.json').read_text(encoding='utf-8'))
    base=json.loads((root/protocol['base_config']).read_text(encoding='utf-8'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'configs').mkdir();manifest=[]
    for condition in protocol['conditions']:
        for seed in protocol['training_seeds']:
            for mode in protocol['training_modes']:
                name=f'{condition}_{mode}_seed{seed}'
                config=dict(base,seed=seed,corruption_seed=protocol['corruption_seed'],
                    training_mode=mode,input_condition=condition,selector_epochs=protocol['selector_epochs'])
                if condition=='reference':config.update(noise_std=0.0,outlier_rate=0.0)
                path=out/'configs'/f'{name}.json';path.write_text(json.dumps(config,indent=2),encoding='utf-8')
                print(f'Running {name}',flush=True)
                with (out/f'{name}.log').open('w',encoding='utf-8') as log:
                    result=subprocess.run([sys.executable,'-X','utf8','-m','ssa_task_srd','--config',str(path),'--out',str(out/name)],
                        cwd=root,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT)
                manifest.append({'condition':condition,'seed':seed,'training_mode':mode,'directory':name,'exit_code':result.returncode})
                (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                if result.returncode:raise RuntimeError(f'{name} failed; inspect its log')
    print(f'Completed {len(manifest)} paired runs: {out}',flush=True)


if __name__=='__main__':main()
