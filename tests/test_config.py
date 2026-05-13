import pytest

from big2_vision_agent.config import Settings
from big2_vision_agent.main import SingleInstanceGuard


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("BIG2_TARGET_URL", raising=False)
    monkeypatch.delenv("BIG2_HEADLESS", raising=False)
    monkeypatch.delenv("BIG2_TIMEOUT_MS", raising=False)
    monkeypatch.setenv("BIG2_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("BIG2_PROFILE_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("BIG2_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BIG2_LOCK_DIR", str(tmp_path / "locks"))

    settings = Settings.from_env()

    assert settings.target_url.startswith("https://www.gamesofa.com/bigtwo/")
    assert settings.headless is False
    assert settings.timeout_ms == 30000
    assert settings.state_path.name == "state.json"
    assert settings.profile_dir.name == "profile"
    assert settings.artifact_dir.name == "artifacts"
    assert settings.lock_dir.name == "locks"


def test_single_instance_guard_rejects_existing_live_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "autoplay_agent.lock"
    lock_path.write_text("12345", encoding="utf-8")
    monkeypatch.setattr("big2_vision_agent.main._pid_exists", lambda pid: pid == 12345)

    with pytest.raises(RuntimeError):
        SingleInstanceGuard(lock_path).acquire()
