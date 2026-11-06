from hinglish.config import load_label_mapping
from hinglish.dataset import (
    check_split_overlap,
    clean_and_map_labels,
    compute_data_hash,
    detect_schema,
    load_raw_dataframe,
    validate_dataset,
)


def test_detect_schema_conll(fixtures_dir):
    schema = detect_schema(fixtures_dir / "sample_conll.txt")
    assert schema.kind == "conll_meta"


def test_detect_schema_csv(fixtures_dir):
    schema = detect_schema(fixtures_dir / "sample.csv")
    assert schema.kind == "csv_flat"
    assert schema.text_col == "text"
    assert schema.label_col == "label"


def test_load_conll(fixtures_dir):
    df, schema = load_raw_dataframe(fixtures_dir / "sample_conll.txt")
    assert schema.kind == "conll_meta"
    assert len(df) == 3
    assert df.iloc[0]["raw_label"] == "positive"
    assert "movie" in df.iloc[0]["text"]


def test_load_csv(fixtures_dir):
    df, schema = load_raw_dataframe(fixtures_dir / "sample.csv")
    assert schema.kind == "csv_flat"
    assert len(df) == 5


def test_validate_dataset_flags_missing_and_duplicates(fixtures_dir):
    labels = load_label_mapping()
    df, _ = load_raw_dataframe(fixtures_dir / "sample.csv")
    report = validate_dataset(df, labels)
    assert report.n_missing_text == 1  # row id=4 has empty text
    assert report.n_exact_duplicates == 2  # rows id=1 and id=5 are exact dups
    assert not report.is_clean()


def test_clean_and_map_labels_drops_invalid_rows(fixtures_dir):
    labels = load_label_mapping()
    df, _ = load_raw_dataframe(fixtures_dir / "sample.csv")
    cleaned = clean_and_map_labels(df, labels)
    assert len(cleaned) == 4  # the empty-text row is dropped
    assert set(cleaned["label_id"].unique()) <= {0, 1, 2}


def test_compute_data_hash_is_stable_to_row_order(fixtures_dir):
    labels = load_label_mapping()
    df, _ = load_raw_dataframe(fixtures_dir / "sample.csv")
    df = clean_and_map_labels(df, labels)
    h1 = compute_data_hash(df)
    h2 = compute_data_hash(df.sample(frac=1, random_state=1).reset_index(drop=True))
    assert h1 == h2


def test_split_overlap_detection(fixtures_dir):
    df, _ = load_raw_dataframe(fixtures_dir / "sample.csv")
    overlap = check_split_overlap(df, df, df)
    assert overlap["train_test_overlap"] > 0  # same df passed 3x on purpose
