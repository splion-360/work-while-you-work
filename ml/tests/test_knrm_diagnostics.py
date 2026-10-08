import numpy as np

from resume_jd_scoring.evaluate import cross_set_neighbor_profile, multiclass_eta_squared


def test_eta_squared_identifies_class_separating_feature():
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    features = np.column_stack((labels.astype(float), np.ones(len(labels))))

    scores = multiclass_eta_squared(features, labels)

    assert scores[0] == 1.0
    assert scores[1] == 0.0


def test_cross_set_neighbor_profile_reports_class_conditioned_composition():
    reference_features = np.asarray([[0.0], [0.1], [10.0], [10.1], [20.0], [20.1]])
    reference_labels = np.asarray([0, 0, 1, 1, 2, 2])
    query_features = np.asarray([[0.05], [10.05], [20.05]])
    query_labels = np.asarray([0, 1, 2])

    profile = cross_set_neighbor_profile(
        reference_features,
        reference_labels,
        query_features,
        query_labels,
        neighbors=2,
        class_count=3,
    )

    assert profile["overall_purity"] == 1.0
    np.testing.assert_array_equal(profile["same_label_fraction"], np.ones(3))
    np.testing.assert_array_equal(profile["neighbor_class_composition"], np.eye(3))
