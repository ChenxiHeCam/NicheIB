from nicheib.eval import deletion_curve, insertion_curve


def test_fidelity_metric_summaries():
    deletion = deletion_curve([0.9, 0.1], base_score=1.0, scores_after_deletion=[1.0, 0.7, 0.4])
    insertion = insertion_curve([0.2, 0.6, 0.9])
    assert deletion["drop"] == 0.6
    assert insertion["gain"] == 0.7
    assert "auc" in deletion
    assert "auc" in insertion
