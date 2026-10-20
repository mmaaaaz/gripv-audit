"""Data access for one GRIP-V category: questions + per-clip params.

Params live in the video parquet files (one per category). Reading the whole
file is wasteful, so:

1. use a local copy if ``data/videos/<category>.parquet`` exists;
2. otherwise read only ``(sample_id, params)`` over HTTPS with DuckDB;
3. cache the result as JSON under ``cache/``.

Both ``data/`` and ``cache/`` are gitignored. No dataset content belongs in git.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import env

CACHE = env.CACHE_DIR
CACHE.mkdir(parents=True, exist_ok=True)
LOCAL_DIR = env.DATA_DIR


def _hf_url(kind: str, category: str) -> str:
    return f"https://huggingface.co/datasets/{env.REPO}/resolve/main/{kind}/{category}.parquet"


def _con():
    import duckdb

    token = env.require_token().replace("'", "''")
    con = duckdb.connect()
    con.execute(f"CREATE SECRET hf (TYPE HTTP, BEARER_TOKEN '{token}')")
    return con


class Store:
    def __init__(self, category: str):
        self.category = category

    # ---------- questions ----------
    def questions(self) -> pd.DataFrame:
        local = LOCAL_DIR / "questions" / f"{self.category}.parquet"
        if local.exists():
            return pd.read_parquet(local)
        return _con().execute(
            f"SELECT * FROM read_parquet('{_hf_url('questions', self.category)}')"
        ).df()

    # ---------- params ----------
    def params(self, refresh: bool = False) -> dict[str, dict]:
        cache = CACHE / f"{self.category}.params.json"
        if cache.exists() and not refresh:
            import json

            return json.loads(cache.read_text(encoding="utf-8"))

        local = LOCAL_DIR / "videos" / f"{self.category}.parquet"
        if local.exists():
            df = pd.read_parquet(local, columns=["sample_id", "params"])
        else:
            df = _con().execute(
                f"SELECT sample_id, params FROM read_parquet('{_hf_url('videos', self.category)}')"
            ).df()

        import json

        out: dict[str, dict] = {}
        for sid, raw in zip(df["sample_id"], df["params"]):
            out[str(sid)] = json.loads(raw) if isinstance(raw, str) else raw
        cache.write_text(json.dumps(out), encoding="utf-8")
        return out
