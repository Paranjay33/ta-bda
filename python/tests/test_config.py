from hinglish.config import load_config, load_label_mapping


def test_label_mapping_loads_and_is_contiguous():
    labels = load_label_mapping()
    assert labels.label_to_id == {"negative": 0, "neutral": 1, "positive": 2}
    assert labels.num_classes == 3
    assert labels.id_to_label == {0: "negative", 1: "neutral", 2: "positive"}


def test_label_mapping_case_insensitive():
    labels = load_label_mapping()
    assert labels.to_id("Positive") == 2
    assert labels.is_valid_label("NEGATIVE")


def test_project_config_loads():
    cfg = load_config()
    assert cfg.get("seed") == 42
    assert cfg.get("transformer", "max_seq_length") == 128
    models_dir = cfg.path("paths", "models_dir")
    assert models_dir.name == "models"
