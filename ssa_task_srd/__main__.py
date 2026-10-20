import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import argparse
import json
from pathlib import Path
from .experiment import run


def main():
    parser=argparse.ArgumentParser(description='Exploratory SSA and seasonal-repair experiment')
    parser.add_argument('--config',default='configs/smoke.json')
    parser.add_argument('--out',required=True,help='New output directory; existing runs are never overwritten')
    parser.add_argument('--fixed-only',action='store_true',help='Skip optional TensorFlow selector stage')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    config_path=Path(args.config)
    if not config_path.is_absolute():config_path=root/config_path
    config=json.loads(config_path.read_text(encoding='utf-8'))
    out=Path(args.out)
    if not out.is_absolute():out=root/out
    run(config,root,out,task_driven=not args.fixed_only)


if __name__=='__main__':main()
