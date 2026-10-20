"""Configuration for the GRIP-V replay harness.

Precedence: environment variables > a local ``.env`` file. The ``.env`` file is
gitignored; never commit a token.

    GRIPV_ENV           path to an alternate env file
    HF_TOKEN            Hugging Face token with read access to the dataset
    GRIPV_REPO          dataset repo id (default: hamzahanif/GRIP-V)
    GRIPV_CACHE         params cache directory (default: ./cache)
    GRIPV_LOCAL_DIR     local questions/videos directory (default: ./data)
    GRIPV_RESULTS       results directory (default: ./results)
"""
from __future__ import annotations

import os
from pathlib import Path


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


ROOT = Path(__file__).resolve().parent.parent


def _discover() -> dict[str, str]:
    candidates = [
        os.environ.get("GRIPV_ENV"),
        Path.cwd() / ".env",
        ROOT / ".env",
    ]
    for candidate in candidates:
        if candidate:
            values = _read_dotenv(Path(candidate).expanduser())
            if values:
                return values
    return {}


_DOTENV = _discover()


def get(key: str, default: str = "") -> str:
    return os.environ.get(key) or _DOTENV.get(key, default)


HF_TOKEN = get("HF_TOKEN")
REPO = get("GRIPV_REPO", "hamzahanif/GRIP-V")

CACHE_DIR = Path(get("GRIPV_CACHE", ROOT / "cache")).expanduser()
DATA_DIR = Path(get("GRIPV_LOCAL_DIR", ROOT / "data")).expanduser()
RESULTS_DIR = Path(get("GRIPV_RESULTS", ROOT / "results")).expanduser()


def require_token() -> str:
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN is not set.\n"
            "Copy .env.example to .env and add a Hugging Face token with read access,\n"
            "or export HF_TOKEN in your shell. The dataset is private; request access first."
        )
    return HF_TOKEN
