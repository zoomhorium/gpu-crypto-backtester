from detector import normalize
import numpy as np

def test_normalize():
    features = np.array([
        [10.0, 1000.0],
        [20.0, 2000.0],
        [30.0, 3000.0],
        [40.0, 4000.0],
    ])
    mean = features.mean(axis=0)
    std  = features.std(axis=0)
    out = normalize(features, mean, std)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-9), "normalized mean must be ~0"
    assert np.allclose(out.std(axis=0), 1, atol=1e-9),  "normalized std must be ~1"

def test_normalize_uses_given_stats():
    # normalizing with EXTERNAL stats (the train/val rule) shifts correctly
    features = np.array([[10.0], [20.0]])
    out = normalize(features, mean=np.array([0.0]), std=np.array([10.0]))
    assert np.allclose(out, [[1.0], [2.0]]), "must use the passed-in mean/std, not its own"

if __name__ == "__main__":
    test_normalize()
    test_normalize_uses_given_stats()
    print("ALL TESTS PASSED")