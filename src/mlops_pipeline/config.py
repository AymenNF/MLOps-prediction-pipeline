from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: Path | str = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
