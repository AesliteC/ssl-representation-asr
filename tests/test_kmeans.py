import sys
import unittest
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_asr.kmeans import fit_kmeans, predict_units, sample_frames


class KMeansTests(unittest.TestCase):
    def test_sample_frames_is_deterministic_and_caps_count(self):
        features = [
            np.arange(12, dtype=np.float32).reshape(6, 2),
            np.arange(8, dtype=np.float32).reshape(4, 2),
        ]

        first = sample_frames(features, max_frames=5, seed=42)
        second = sample_frames(features, max_frames=5, seed=42)

        self.assertEqual(first.shape, (5, 2))
        self.assertTrue(np.array_equal(first, second))

    def test_fit_kmeans_and_predict_units(self):
        frames = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 10.0],
                [10.1, 10.0],
            ],
            dtype=np.float32,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="MiniBatchKMeans is known to have a memory leak")
            model = fit_kmeans(frames, codebook_size=2, seed=0, batch_size=4096, max_iter=20)
        units = predict_units(model, frames)

        self.assertEqual(model.cluster_centers_.shape, (2, 2))
        self.assertEqual(units.shape, (4,))
        self.assertTrue(set(units.tolist()) <= {0, 1})


if __name__ == "__main__":
    unittest.main()
