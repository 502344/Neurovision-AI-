import numpy as np
import pytest

from ml.fusion.feature_fusion import fuse_features


def test_feature_fusion_concat():
    a, b = np.random.rand(5, 8), np.random.rand(5, 3)
    f = fuse_features(a, b, "concat")
    assert f.shape == (5, 11)
    np.testing.assert_allclose(f[:, :8], a, rtol=1e-5)


@pytest.mark.parametrize("m", ["l2_concat", "weighted_concat"])
def test_other_fusion_methods(m):
    f = fuse_features(np.random.rand(4, 6), np.random.rand(4, 2), m)
    assert f.shape == (4, 8) and np.isfinite(f).all()


def test_fusion_errors():
    with pytest.raises(ValueError):
        fuse_features(np.ones((3, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError):
        fuse_features(np.ones((3, 2)), np.ones((3, 2)), "nope")
