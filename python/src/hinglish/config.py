"""Central configuration loader.

Both the Python/Transformer branch and the Scala/Spark branch read the SAME
``configs/label_mapping.json`` file, so the label -> id mapping never drifts
between branches. General hyperparameters/paths live in ``configs/config.yaml``.

Do not hard-code the project root. Resolution order:
    1. ``HINGLISH_PROJECT_ROOT`` environment variable, if set.
    2. Walk up from this file until a directory containing ``configs/config.yaml``
       is found.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def find_project_root() -> Path:
    env_root = os.environ.get("HINGLISH_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if not (root / "configs" / "config.yaml").exists():
            raise FileNotFoundError(
                f"HINGLISH_PROJECT_ROOT={root} does not contain configs/config.yaml"
            )
        return root

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "configs" / "config.yaml").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate project root (no configs/config.yaml found in any "
        "parent directory). Set HINGLISH_PROJECT_ROOT explicitly."
    )


@dataclass(frozen=True)
class LabelMapping:
    label_to_id: dict[str, int]
    # Optional raw-label aliases (e.g. the supplied SentiMix files use "0"/"1"/"2").
    # Aliases map a raw dataset value -> a canonical label NAME; they never
    # introduce a second name->id mapping (label_mapping.json stays canonical).
    aliases: dict[str, str] = field(default_factory=dict)

    @property
    def id_to_label(self) -> dict[int, str]:
        return {v: k for k, v in self.label_to_id.items()}

    @property
    def num_classes(self) -> int:
        return len(self.label_to_id)

    @property
    def ordered_names(self) -> list[str]:
        return [self.id_to_label[i] for i in range(self.num_classes)]

    def canonical_name(self, label: str) -> str | None:
        key = str(label).strip().lower()
        if key in self.label_to_id:
            return key
        alias = self.aliases.get(key)
        if alias is not None and alias in self.label_to_id:
            return alias
        return None

    def is_valid_label(self, label: str) -> bool:
        return self.canonical_name(label) is not None

    def to_id(self, label: str) -> int:
        key = self.canonical_name(label)
        if key is None:
            raise ValueError(
                f"Unknown label {label!r}. Valid labels: {sorted(self.label_to_id)}"
            )
        return self.label_to_id[key]


def load_label_mapping(root: Path | None = None, aliases: dict | None = None) -> LabelMapping:
    root = root or find_project_root()
    path = root / "configs" / "label_mapping.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    normalized = {str(k).strip().lower(): int(v) for k, v in raw.items()}
    ids = sorted(normalized.values())
    if ids != list(range(len(ids))):
        raise ValueError(
            f"Label mapping ids in {path} must be a contiguous 0..N-1 range, got {ids}"
        )
    alias_map = {str(k).strip().lower(): str(v).strip().lower() for k, v in (aliases or {}).items()}
    bad = {k: v for k, v in alias_map.items() if v not in normalized}
    if bad:
        raise ValueError(f"raw_label_aliases point at unknown canonical labels: {bad}")
    return LabelMapping(label_to_id=normalized, aliases=alias_map)


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    raw: dict[str, Any]
    labels: LabelMapping

    def path(self, *keys: str) -> Path:
        """Resolve a dotted path config value (e.g. 'paths', 'raw_data_dir') to
        an absolute Path under the project root."""
        node: Any = self.raw
        for k in keys:
            node = node[k]
        return (self.root / str(node)).resolve()

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _coerce(value: str) -> Any:
    # int/float first: PyYAML (YAML 1.1) reads "1e-5" as a STRING because it lacks a dot.
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    parsed = yaml.safe_load(value)
    return parsed if parsed is not None else value


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply `section.key=value` overrides (values parsed as YAML scalars)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = raw
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _coerce(value.strip())
    return raw


def load_config(root: Path | None = None, overrides: list[str] | None = None) -> ProjectConfig:
    """Load configs/config.yaml, then (optionally) the overlay named by the
    HINGLISH_CONFIG_OVERLAY env var (e.g. configs/kaggle_v3.yaml), then any
    `section.key=value` overrides. Every effective value ends up in the run
    manifest, so a run is always reproducible from its manifest."""
    root = root or find_project_root()
    with open(root / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    overlay = os.environ.get("HINGLISH_CONFIG_OVERLAY")
    if overlay:
        p = Path(overlay) if Path(overlay).is_absolute() else root / overlay
        with open(p, "r", encoding="utf-8") as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})
    raw = apply_overrides(raw, overrides or [])
    aliases = ((raw or {}).get("dataset") or {}).get("raw_label_aliases") or {}
    labels = load_label_mapping(root, aliases=aliases)
    return ProjectConfig(root=root, raw=raw, labels=labels)
