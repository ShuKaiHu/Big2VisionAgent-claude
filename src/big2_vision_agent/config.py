from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TARGET_URL = "https://www.gamesofa.com/bigtwo/#"


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    target_url: str
    headless: bool
    timeout_ms: int
    state_path: Path
    profile_dir: Path
    artifact_dir: Path
    lock_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            target_url=os.getenv("BIG2_TARGET_URL", DEFAULT_TARGET_URL),
            headless=_read_bool("BIG2_HEADLESS", False),
            timeout_ms=int(os.getenv("BIG2_TIMEOUT_MS", "30000")),
            state_path=Path(
                os.getenv("BIG2_STATE_PATH", "state/storage_state.json")
            ).expanduser(),
            profile_dir=Path(
                os.getenv("BIG2_PROFILE_DIR", "state/browser-profile")
            ).expanduser(),
            artifact_dir=Path(
                os.getenv("BIG2_ARTIFACT_DIR", "artifacts")
            ).expanduser(),
            lock_dir=Path(
                os.getenv("BIG2_LOCK_DIR", "state/locks")
            ).expanduser(),
        )
