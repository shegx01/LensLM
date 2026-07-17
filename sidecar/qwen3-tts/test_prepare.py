"""Offline unit tests for `--prepare` mode — no MLX, no real download (#194).

`snapshot_download` and the HF metadata call are mocked; the module import path
is the same one `test_resolve_gen_params.py` uses (MLX stays deferred inside
`load_qwen`), so these run at macOS signoff without MLX installed.
"""

import os
import threading

from qwen3_tts_sidecar import (
    MODEL_ID,
    dir_size_bytes,
    model_cache_dir,
    parse_prepare,
    run_prepare,
    serve,
    _poll_progress,
)


def test_parse_prepare_arg():
    assert parse_prepare(["--prepare"]) is True
    assert parse_prepare(["python", "qwen3_tts_sidecar.py", "--prepare"]) is True
    assert parse_prepare([]) is False
    assert parse_prepare(["slow"]) is False


def test_serve_mode_symbol_is_unaffected():
    # Serve mode must remain the callable default entry (renamed from `main`),
    # and importing this module must not have pulled in MLX.
    assert callable(serve)
    import sys

    assert "mlx" not in sys.modules


def test_dir_size_sums_all_files_including_incomplete(tmp_path):
    blobs = tmp_path / "hub" / "models--x" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "a").write_bytes(b"x" * 100)
    (blobs / "b.incomplete").write_bytes(b"y" * 50)
    (tmp_path / "hub" / "models--x" / "snapshots").mkdir()
    assert dir_size_bytes(str(tmp_path)) == 150


def test_dir_size_missing_dir_is_zero(tmp_path):
    assert dir_size_bytes(str(tmp_path / "nope")) == 0


def test_model_cache_dir_honors_hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    expected = os.path.join(
        str(tmp_path),
        "hub",
        "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    )
    assert model_cache_dir() == expected


def test_poll_emits_progress_shape(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "blob").write_bytes(b"z" * (3 * 1024 * 1024))
    emitted = []
    stop = threading.Event()
    t = threading.Thread(
        target=_poll_progress,
        args=(str(cache), 4096, stop, emitted.append, 0.01),
    )
    t.start()
    # Give the poller a couple of intervals to observe the file, then stop.
    import time

    time.sleep(0.1)
    stop.set()
    t.join(timeout=2)

    assert emitted, "poller emitted at least one progress line"
    msg = emitted[0]
    assert set(msg) == {"progress"}
    assert set(msg["progress"]) == {"received", "total"}
    assert msg["progress"]["received"] == 3 * 1024 * 1024
    assert msg["progress"]["total"] == 4096


class _FakeSibling:
    def __init__(self, size):
        self.size = size


class _FakeInfo:
    def __init__(self, sizes):
        self.siblings = [_FakeSibling(s) for s in sizes]


class _FakeApi:
    def model_info(self, model_id, files_metadata=False):
        assert model_id == MODEL_ID
        assert files_metadata is True
        return _FakeInfo([1000, 2000, 3000])


def _make_fake_hf(tmp_path, monkeypatch, snapshot_impl):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    class _FakeHf:
        HfApi = _FakeApi
        snapshot_download = staticmethod(snapshot_impl)

    return _FakeHf()


def test_run_prepare_success_emits_done_last(tmp_path, monkeypatch):
    def fake_snapshot(model_id):
        assert model_id == MODEL_ID
        # Simulate the download writing a blob into the model cache dir.
        target = os.path.join(model_cache_dir(), "blobs")
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "weights"), "wb") as f:
            f.write(b"w" * 6000)

    hf = _make_fake_hf(tmp_path, monkeypatch, fake_snapshot)
    emitted = []
    code = run_prepare(hf=hf, emit=emitted.append, interval=0.01)

    assert code == 0
    assert emitted[-1] == {"done": True}
    # The deterministic final tick reports the real on-disk size + summed total.
    final_progress = emitted[-2]
    assert final_progress == {"progress": {"received": 6000, "total": 6000}}


def test_run_prepare_error_emits_error_and_nonzero(tmp_path, monkeypatch):
    def fake_snapshot(model_id):
        raise RuntimeError("network exploded")

    hf = _make_fake_hf(tmp_path, monkeypatch, fake_snapshot)
    emitted = []
    code = run_prepare(hf=hf, emit=emitted.append, interval=0.01)

    assert code == 1
    assert emitted[-1]["error"].startswith("network exploded")
    assert not any("done" in m for m in emitted)


def test_run_prepare_indeterminate_total_when_metadata_fails(tmp_path, monkeypatch):
    class _BrokenApi:
        def model_info(self, *a, **k):
            raise RuntimeError("no network")

    def fake_snapshot(model_id):
        pass

    monkeypatch.setenv("HF_HOME", str(tmp_path))

    class _FakeHf:
        HfApi = _BrokenApi
        snapshot_download = staticmethod(fake_snapshot)

    emitted = []
    code = run_prepare(hf=_FakeHf(), emit=emitted.append, interval=0.01)

    assert code == 0
    assert emitted[-1] == {"done": True}
    assert emitted[-2]["progress"]["total"] is None
