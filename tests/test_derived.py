"""Die gespeicherte Projektion rechnet wie sklearn -- sonst landet ein neues Bild im falschen Raum."""

import numpy as np
import pytest

from src.models.derived import load_projection, project

sklearn = pytest.importorskip("sklearn.decomposition")


@pytest.mark.parametrize("whiten", [False, True])
def test_project_wie_sklearn(tmp_path, whiten):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 32)).astype(np.float32) @ rng.standard_normal((32, 32)).astype(np.float32)
    pca = sklearn.PCA(n_components=8, whiten=whiten, svd_solver="full").fit(x)
    np.savez(tmp_path / "p.npz", mean=pca.mean_, components=pca.components_,
             explained_variance=pca.explained_variance_, whiten=np.array(whiten))
    proj = load_projection(tmp_path / "p.npz")
    neu = rng.standard_normal((20, 32)).astype(np.float32)
    soll = pca.transform(neu)
    soll = soll / np.linalg.norm(soll, axis=1, keepdims=True)
    assert np.allclose(project(neu, proj), soll, atol=1e-5)
