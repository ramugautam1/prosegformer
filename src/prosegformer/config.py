"""YAML configurations with command-line overrides (`--set optim.steps=500 seed=3`)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(defaults: dict, path: Path | None = None, overrides: list[str] = ()) -> dict:
    cfg = json.loads(json.dumps(defaults))
    for k, v in ((yaml.safe_load(Path(path).read_text()) if path else None) or {}).items():
        cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return cfg
