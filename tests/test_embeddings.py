# tests/test_embeddings.py
import numpy as np

from app.embeddings import embed_texts


def test_embed_shape_and_norm():
    vecs = embed_texts(["猫是哺乳动物", "今天天气不错"])
    arr = np.array(vecs)
    assert arr.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(arr, axis=-1), 1.0, atol=1e-3)


def test_semantic_similarity_sanity():
    vecs = np.array(embed_texts(["小猫在睡觉", "猫科动物", "股市大涨"]))
    sim_cat = vecs[0] @ vecs[1]
    sim_unrelated = vecs[0] @ vecs[2]
    assert sim_cat > sim_unrelated
