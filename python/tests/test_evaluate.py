from hinglish.evaluate import compute_classification_metrics


def test_perfect_predictions_give_f1_of_one():
    id_to_label = {0: "negative", 1: "neutral", 2: "positive"}
    y = [0, 1, 2, 0, 1, 2]
    m = compute_classification_metrics(y, y, id_to_label)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.weighted_f1 == 1.0
    assert sum(sum(row) for row in m.confusion_matrix) == len(y)


def test_confusion_matrix_shape_matches_num_classes():
    id_to_label = {0: "negative", 1: "neutral", 2: "positive"}
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 0]
    m = compute_classification_metrics(y_true, y_pred, id_to_label)
    assert len(m.confusion_matrix) == 3
    assert all(len(row) == 3 for row in m.confusion_matrix)
    assert 0.0 <= m.macro_f1 <= 1.0
    assert set(m.per_class.keys()) == {"negative", "neutral", "positive"}
