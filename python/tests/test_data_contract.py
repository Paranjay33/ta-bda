"""Tests for the shared processed-data contract, label aliases, the
headerless-TSV schema, the dataset-style serving transform and calibration.

Fixture data here is tiny and synthetic: SOFTWARE TESTS ONLY, not benchmark data.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hinglish.calibration import (expected_calibration_error, fit_margin_platt, fit_temperature, nll,
                                  probabilistic_metrics, softmax)
from hinglish.config import find_project_root, load_config, load_label_mapping
from hinglish.dataset import detect_schema, load_raw_dataframe
from hinglish.dataset_style import DatasetStyleConfig, to_dataset_style
from hinglish.normalization import NormalizationConfig, normalize_text
from hinglish.prepare_data import build_split, strip_nan_artifact

ROOT = find_project_root()


# ------------------------------------------------------------------ labels
def test_raw_label_aliases_map_to_canonical_ids():
    labels = load_label_mapping(aliases={"0": "negative", "1": "neutral", "2": "positive"})
    assert labels.to_id("0") == 0 and labels.to_id("1") == 1 and labels.to_id("2") == 2
    assert labels.to_id("Positive") == 2
    assert labels.ordered_names == ["negative", "neutral", "positive"]
    assert not labels.is_valid_label("3")


def test_alias_to_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        load_label_mapping(aliases={"0": "very_negative"})


def test_project_config_carries_dataset_aliases():
    cfg = load_config()
    assert cfg.labels.to_id("0") == 0
    assert cfg.labels.label_to_id == {"negative": 0, "neutral": 1, "positive": 2}


# ------------------------------------------------------------------ schema
def _write(p: Path, lines: list[str]) -> Path:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_headerless_uid_text_label_tsv_detected_and_parsed(tmp_path):
    f = _write(tmp_path / "t.tsv", ["11\tyeh movi acchi thi\t2", "12\tbakwa film nan\t0", "13\tkal offic jana\t1"])
    assert detect_schema(f).kind == "tsv_uid_text_label"
    df, _ = load_raw_dataframe(f)
    assert df["id"].tolist() == ["11", "12", "13"]
    assert df["raw_label"].tolist() == ["2", "0", "1"]


def test_unlabelled_tsv_detected(tmp_path):
    f = _write(tmp_path / "u.tsv", ["1\tkuch bhi", "2\tsab theek"])
    assert detect_schema(f).kind == "tsv_uid_text"


def test_unknown_schema_raises(tmp_path):
    f = _write(tmp_path / "x.txt", ["just some text", "no structure here"])
    with pytest.raises(ValueError):
        load_raw_dataframe(f)


# ------------------------------------------------------------------ nan artefact
@pytest.mark.parametrize("raw,expected", [
    ("acchi film nan", "acchi film"),
    ("nan", ""),
    ("nanak ji ki jai", "nanak ji ki jai"),
    ("banana", "banana"),
    ("kuch nahi nan  ", "kuch nahi"),
])
def test_strip_nan_artifact(raw, expected):
    assert strip_nan_artifact(raw) == expected


# ------------------------------------------------------------------ split building
def _fixture_project(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "label_mapping.json", root / "configs")
    shutil.copytree(ROOT / "configs" / "dataset_compat", root / "configs" / "dataset_compat")
    cfg_text = (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    (root / "configs" / "config.yaml").write_text(cfg_text, encoding="utf-8")
    raw = root / "data" / "raw" / "sentimix_supplied"
    raw.mkdir(parents=True)
    half = len(rows) // 2
    _write(raw / "FinalTrainingOnly.tsv", [f"{a}\t{b}\t{c}" for a, b, c in rows[:half]])
    _write(raw / "ValidationOnly.tsv", [f"{a}\t{b}\t{c}" for a, b, c in rows[half:]])
    return root


def _rows(n=120):
    rng = np.random.default_rng(0)
    words = ["acchi", "bakwa", "movi", "thi", "yaar", "kal", "offic", "bahut", "badhia", "nahi"]
    return [(str(i), " ".join(rng.choice(words, 5)) + f" w{i}", str(i % 3)) for i in range(n)]


def test_build_split_is_deterministic_disjoint_and_dedups(tmp_path):
    rows = _rows(120)
    rows.append(("900", rows[0][1], rows[0][2]))          # exact duplicate, same label -> collapsed
    rows.append(("901", rows[1][1] + " nan", rows[1][2]))  # same text after nan-strip -> collapsed
    rows.append(("902", "conflict text", "0"))
    rows.append(("903", "conflict text", "2"))             # conflicting labels -> dropped entirely
    rows.append(("904", "nan", "1"))                       # empty after cleaning -> dropped
    root = _fixture_project(tmp_path, rows)
    cfg = load_config(root)
    s1, m1 = build_split(cfg)
    s2, m2 = build_split(cfg)
    for k in ("train", "validation", "test"):
        assert m1["splits"][k]["sample_ids_sha256"] == m2["splits"][k]["sample_ids_sha256"]
    assert all(v == 0 for v in m1["overlap_checks_normalized_text"].values())
    c = m1["counts"]
    assert c["dropped_empty_after_normalization"] == 1
    assert c["conflicting_label_texts"] == 1 and c["dropped_conflicting_label_rows"] == 2
    assert c["collapsed_exact_duplicates"] == 2
    all_text = pd.concat(s1.values())["normalized_text"]
    assert all_text.is_unique and "conflict text" not in set(all_text)
    total = sum(len(v) for v in s1.values())
    assert total == c["final_unique_samples"] == 120
    assert abs(len(s1["test"]) - 12) <= 1


def test_processed_label_ids_follow_canonical_mapping(tmp_path):
    root = _fixture_project(tmp_path, _rows(60))
    splits, _ = build_split(load_config(root))
    df = pd.concat(splits.values())
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    assert (df["label"].map(mapping) == df["label_id"]).all()


# ------------------------------------------------------------------ real frozen split (integration)
@pytest.mark.skipif(not (ROOT / "data" / "processed" / "split_manifest.json").exists(), reason="split not built")
def test_frozen_split_files_match_manifest_and_contract():
    from hinglish.prepare_data import load_processed_split, load_split_manifest, sha256_file

    cfg = load_config()
    m = load_split_manifest(cfg)
    norm_cfg = NormalizationConfig.from_project_config(cfg.get("preprocessing"))
    seen = set()
    for name in ("train", "validation", "test"):
        assert sha256_file(cfg.root / m["splits"][name]["file"]) == m["splits"][name]["file_sha256"]
        df = load_processed_split(cfg, name)
        assert len(df) == m["splits"][name]["rows"]
        texts = set(df["normalized_text"])
        assert not (texts & seen), f"{name} overlaps an earlier split"
        seen |= texts
        # Cross-branch contract: normalized_text == normalize(strip_nan(raw_text)).
        sample = df.sample(min(300, len(df)), random_state=0)
        for raw, normed in zip(sample["raw_text"], sample["normalized_text"]):
            assert normalize_text(strip_nan_artifact(raw), norm_cfg) == normed


# ------------------------------------------------------------------ dataset-style transform
@pytest.fixture(scope="module")
def style():
    return DatasetStyleConfig.from_project_config(load_config())


def test_dataset_style_matches_training_space(style):
    out = to_dataset_style("Movie toh acchi thi but ending bakwas thi", style)
    assert out == "movi toh acchi thi end bakwa thi"  # 'but' is an NLTK stopword; Porter stems


def test_dataset_style_preserves_placeholders_emoji_and_hindi_negation(style):
    norm = normalize_text("@rahul check http://x.com yeh bilkul accha nahi 😡!!!", NormalizationConfig())
    out = to_dataset_style(norm, style)
    assert "<USER>" in out and "<URL>" in out
    assert "nahi" in out and "😡" in out and "!" in out


def test_dataset_style_splits_contractions_like_training_data(style):
    assert "n't" in to_dataset_style("I don't like it", style).split()


def test_dataset_style_is_near_idempotent_on_training_text(style):
    cfg = load_config()
    p = cfg.root / "data" / "processed" / "validation.tsv"
    if not p.exists():
        pytest.skip("split not built")
    from hinglish.prepare_data import load_processed_split

    df = load_processed_split(cfg, "validation").head(400)
    toks = same = 0
    for t in df["normalized_text"]:
        a, b = t.split(), to_dataset_style(t, style).split()
        toks += len(a)
        same += len(set(a) & set(b))
    assert same / toks > 0.85


# ------------------------------------------------------------------ calibration
def test_temperature_scaling_reduces_nll_and_preserves_argmax():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    logits = rng.normal(size=(600, 3))
    logits[np.arange(600), y] += 1.0
    logits *= 4.0  # over-confident
    cal = fit_temperature(logits, y)
    assert cal.temperature > 1.0
    assert cal.val_nll_after <= cal.val_nll_before
    assert (cal.apply(logits).argmax(1) == logits.argmax(1)).all()
    assert np.allclose(cal.apply(logits).sum(1), 1.0)
    assert cal.apply(logits[0]).shape == (1, 3)


def test_platt_on_margins_outputs_distributions():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    m = rng.normal(size=(300, 3))
    m[np.arange(300), y] += 1.5
    cal = fit_margin_platt(m, y)
    p = cal.apply(m)
    assert p.shape == (300, 3) and np.allclose(p.sum(1), 1.0)
    assert nll(p, y) < nll(np.full((300, 3), 1 / 3), y)


def test_probabilistic_metrics_bounds():
    y = np.array([0, 1, 2, 2])
    perfect = np.eye(3)[y]
    pm = probabilistic_metrics(perfect, y)
    assert pm["ece_15_bins"] == pytest.approx(0.0, abs=1e-9)
    assert pm["brier_multiclass"] == pytest.approx(0.0)
    assert 0.0 <= expected_calibration_error(softmax(np.zeros((4, 3))), y) <= 1.0


def test_label_mapping_file_is_the_single_source():
    data = json.loads((ROOT / "configs" / "label_mapping.json").read_text())
    assert data == {"negative": 0, "neutral": 1, "positive": 2}
    scala = (ROOT / "spark" / "src" / "main" / "scala" / "hinglish" / "config" / "Config.scala").read_text()
    assert "configs/label_mapping.json" in scala
