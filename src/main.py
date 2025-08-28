import os
import argparse
import yaml

from .train import run_experiment1, run_experiment2, run_experiment3


def load_config(path: str):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='ASTME Synthetic Experiments Runner')
    parser.add_argument('--config', type=str, default='config/astme_synth.yaml', help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    fast = bool(cfg.get('fast', True))
    out_root = cfg.get('output_dir', '.research/iteration1/images')

    exp1_dir = os.path.join(out_root, 'exp1')
    exp2_dir = os.path.join(out_root, 'exp2')
    exp3_dir = os.path.join(out_root, 'exp3')

    os.makedirs(exp1_dir, exist_ok=True)
    os.makedirs(exp2_dir, exist_ok=True)
    os.makedirs(exp3_dir, exist_ok=True)

    print('[MAIN] Running experiments with fast=%s' % str(fast))
    run_experiment1(save_dir=exp1_dir, fast=fast)
    run_experiment2(save_dir=exp2_dir, fast=fast)
    run_experiment3(save_dir=exp3_dir, fast=fast)

    # List generated PDFs
    print('[MAIN] Generated figures:')
    for root, _, files in os.walk(out_root):
        for f in files:
            if f.endswith('.pdf'):
                print('  -', os.path.join(root, f))


if __name__ == '__main__':
    main()
