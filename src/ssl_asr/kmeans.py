"""K-means helpers for discrete SSL units."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def sample_frames(
    feature_arrays: Sequence[np.ndarray],
    *,
    max_frames: int,
    seed: int,
) -> np.ndarray:
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    if not feature_arrays:
        raise ValueError("feature_arrays must not be empty")

    frames = np.concatenate([np.asarray(array, dtype=np.float32) for array in feature_arrays], axis=0)
    if frames.ndim != 2:
        raise ValueError(f"expected 2-D frame arrays, got shape {frames.shape}")
    if len(frames) <= max_frames:
        return frames

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(frames), size=max_frames, replace=False)
    return frames[np.sort(indices)]


def fit_kmeans(
    frames: np.ndarray,
    *,
    codebook_size: int,
    seed: int = 42,
    batch_size: int = 4096,
    max_iter: int = 100,
    n_init: int = 3,
) -> MiniBatchKMeans:
    if codebook_size < 2:
        raise ValueError("codebook_size must be at least 2")
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"expected 2-D frames, got shape {frames.shape}")
    if len(frames) < codebook_size:
        raise ValueError("number of frames must be at least codebook_size")

    model = MiniBatchKMeans(
        n_clusters=codebook_size,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    return model.fit(frames)


def predict_units(model: MiniBatchKMeans, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"expected 2-D features, got shape {features.shape}")
    return model.predict(features).astype(np.int64)
