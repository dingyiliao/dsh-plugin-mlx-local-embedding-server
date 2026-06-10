from app.store import VectorStore


def make_store(tmp_path, dim=4):
    return VectorStore(str(tmp_path / "test.db"), dim=dim)


def test_add_and_count(tmp_path):
    s = make_store(tmp_path)
    n = s.add(["猫", "狗"], [[1, 0, 0, 0], [0, 1, 0, 0]], source="a.txt")
    assert n == 2
    assert s.count() == 2


def test_search_returns_nearest_first(tmp_path):
    s = make_store(tmp_path)
    s.add(
        ["猫", "狗", "小猫"],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]],
        source="a.txt",
    )
    results = s.search([1, 0, 0, 0], top_k=2)
    assert [r["text"] for r in results] == ["猫", "小猫"]
    assert results[0]["distance"] <= results[1]["distance"]
    assert results[0]["source"] == "a.txt"


def test_search_empty_store(tmp_path):
    s = make_store(tmp_path)
    assert s.search([1, 0, 0, 0], top_k=3) == []


def test_persistence(tmp_path):
    path = str(tmp_path / "p.db")
    VectorStore(path, dim=4).add(["猫"], [[1, 0, 0, 0]], source="a")
    s2 = VectorStore(path, dim=4)
    assert s2.count() == 1
