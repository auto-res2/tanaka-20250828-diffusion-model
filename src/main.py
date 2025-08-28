import argparse
import os
import yaml

from .preprocess import prepare_environment
from .train import quick_test, run_experiment_1, run_experiment_2, run_experiment_3
from .evaluate import evaluate_model_checkpoint


def parse_args():
    ap = argparse.ArgumentParser(description="RevLoRA-Diffusion Toy Experiments")
    ap.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    images_dir = cfg.get("paths", {}).get("images_dir", ".research/iteration2/images")
    models_dir = cfg.get("paths", {}).get("models_dir", "models")

    env = prepare_environment(images_dir, models_dir)
    images_dir = env["images_dir"]
    models_dir = env["models_dir"]

    mode = cfg.get("mode", "quick")

    if mode == "quick":
        quick_test(images_dir=images_dir, models_dir=models_dir)
        evaluate_model_checkpoint(models_dir=models_dir, images_dir=images_dir)
    else:
        # granular control
        if cfg.get("experiments", {}).get("exp1", True):
            run_experiment_1(images_dir=images_dir)
        if cfg.get("experiments", {}).get("exp2", True):
            run_experiment_2(images_dir=images_dir, models_dir=models_dir)
        if cfg.get("experiments", {}).get("exp3", True):
            run_experiment_3(images_dir=images_dir)
        if cfg.get("evaluate", True):
            evaluate_model_checkpoint(models_dir=models_dir, images_dir=images_dir)


if __name__ == "__main__":
    main()
