import numpy as np

from env.normalization import StateNormalizer


def test_normalize_preserves_dtype_and_shape():
    normalizer = StateNormalizer(mean=[1.0, 2.0], std=[1.0, 2.0], expected_dim=2)
    raw = np.array([3.0, 6.0], dtype=np.float32)
    out = normalizer.normalize(raw)
    np.testing.assert_allclose(out, [2.0, 2.0], rtol=1e-6, atol=1e-6)
    assert out.dtype == np.float32
    assert out.shape == (2,)
    np.testing.assert_array_equal(raw, np.array([3.0, 6.0], dtype=np.float32))


def test_normalize_clips_in_place():
    normalizer = StateNormalizer(mean=[0.0, 0.0], std=[1.0, 1.0], expected_dim=2, clip_min=-1.0, clip_max=1.0)
    raw = np.array([-5.0, 5.0], dtype=np.float32)
    out = normalizer.normalize(raw)
    np.testing.assert_allclose(out, [-1.0, 1.0], rtol=1e-6, atol=1e-6)
    assert out.dtype == np.float32
