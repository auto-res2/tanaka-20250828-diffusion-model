import os
from typing import Dict

from .train import ensure_dir


def prepare_environment(base_images_dir: str = ".research/iteration2/images", models_dir: str = "models") -> Dict[str, str]:
    ensure_dir(base_images_dir)
    ensure_dir(models_dir)
    return {"images_dir": base_images_dir, "models_dir": models_dir}
