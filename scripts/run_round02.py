"""Locked validation-only ablation; subprocess logs and outputs never overwrite runs."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/round02_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute(): out = root/out
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root/'configs/round02_protocol.json').read_text(encoding='utf-8'))
    base = json.loads((root/protocol['base_config']).read_text(encoding='utf-8'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'generated_configs').mkdir()
    manifest = []
    for condition in protocol['conditions']:
        for seed in protocol['training_seeds']:
            for mode in protocol['selector_modes']:
                name = f'{condition}_{mode}_seed{seed}'
                config = dict(base, seed=seed, selector_mode=mode, corruption_seed=protocol['corruption_seed'],
                    evaluation_splits=protocol['evaluation_splits'], input_condition=condition)
                if condition == 'reference': config.update(noise_std=0.0, outlier_rate=0.0)
                config_file = out/'generated_configs'/f'{name}.json'
                config_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
                log_file = out/f'{name}.log'
                print(f'Running {name}',flush=True)
                env = dict(os.environ, PYTHONIOENCODING='utf-8')
                with log_file.open('w',encoding='utf-8') as log:
                    result = subprocess.run([sys.executable,'-X','utf8','-m','ssa_task_srd',
                        '--config',str(config_file),'--out',str(out/name)],cwd=root,env=env,
                        stdout=log,stderr=subprocess.STDOUT)
                entry = {'condition':condition,'training_seed':seed,'mode':mode,'directory':name,'exit_code':result.returncode}
                manifest.append(entry)
                (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                if result.returncode:
                    raise RuntimeError(f'{name} failed; inspect {log_file}')
    print(f'Completed {len(manifest)} runs: {out}',flush=True)


if __name__ == '__main__': main()
