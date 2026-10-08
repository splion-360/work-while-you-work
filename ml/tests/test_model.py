import torch

from resume_jd_scoring.model import ResumeJDLinearClassifier


def test_linear_classifier_maps_feature_vectors_to_three_logits():
    model = ResumeJDLinearClassifier(input_dimension=8)

    logits = model(torch.ones(5, 8))

    assert logits.shape == (5, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) == 27


def test_linear_classifier_rejects_invalid_dimensions():
    for kwargs in (
        {"input_dimension": 0},
        {"input_dimension": 8, "class_count": 1},
    ):
        try:
            ResumeJDLinearClassifier(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"configuration should fail: {kwargs}")
