import sqlite3
import threading

import sqlite_vec


class VectorStore:
    """sqlite-vec 向量库：存 (向量, 原文, 来源)，按余弦距离检索。"""

    def __init__(self, path: str, dim: int = 1024):
        self.dim = dim
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute(
            "create table if not exists chunks("
            "id integer primary key autoincrement, "
            "text text not null, source text not null)"
        )
        self.db.execute(
            f"create virtual table if not exists chunks_vec "
            f"using vec0(embedding float[{dim}])"
        )

    def add(self, texts: list[str], vectors: list[list[float]], source: str) -> int:
        with self._lock:
            for text, vec in zip(texts, vectors, strict=True):
                cur = self.db.execute(
                    "insert into chunks(text, source) values (?, ?)", (text, source)
                )
                self.db.execute(
                    "insert into chunks_vec(rowid, embedding) values (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(list(vec))),
                )
            self.db.commit()
        return len(texts)

    def search(self, vector: list[float], top_k: int = 4) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "select c.text, c.source, v.distance "
                "from chunks_vec v join chunks c on c.id = v.rowid "
                "where v.embedding match ? and k = ? order by v.distance",
                (sqlite_vec.serialize_float32(list(vector)), top_k),
            ).fetchall()
        return [
            {"text": t, "source": s, "distance": d} for t, s, d in rows
        ]

    def count(self) -> int:
        with self._lock:
            return self.db.execute("select count(*) from chunks").fetchone()[0]
