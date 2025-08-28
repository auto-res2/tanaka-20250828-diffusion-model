import os
import sys
import json
import yaml

from .preprocess import ensure_dir
from .evaluate import experiment1, experiment2, experiment3


DEFAULT_CONFIG_PATH = os.path.join('config', 'config.yaml')


def run_from_config(cfg_path: str = DEFAULT_CONFIG_PATH):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    out_base = cfg.get('output_base_dir', '.research/iteration5/images')
    ensure_dir(out_base)

    quick = bool(cfg.get('quick', True))
    exps = cfg.get('experiments', [1,2,3])

    results = {}

    if 1 in exps:
        out_dir = os.path.join(out_base, 'exp1')
        ensure_dir(out_dir)
        print(f"Running Experiment 1 -> {out_dir}")
        results['exp1'] = experiment1(output_dir=out_dir, quick=quick)

    if 2 in exps:
        out_dir = os.path.join(out_base, 'exp2')
        ensure_dir(out_dir)
        print(f"Running Experiment 2 -> {out_dir}")
        results['exp2'] = experiment2(output_dir=out_dir, quick=quick)

    if 3 in exps:
        out_dir = os.path.join(out_base, 'exp3')
        ensure_dir(out_dir)
        print(f"Running Experiment 3 -> {out_dir}")
        results['exp3'] = experiment3(output_dir=out_dir, quick=quick)

    print("All experiments finished.")
    print(json.dumps(results, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x)))


if __name__ == '__main__':
    cfg_path = DEFAULT_CONFIG_PATH
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    run_from_config(cfg_path)
