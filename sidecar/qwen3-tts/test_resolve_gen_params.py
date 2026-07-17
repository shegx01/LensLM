"""Offline unit tests for `resolve_gen_params` — no MLX/model, run at macOS signoff.

Imports the sidecar module directly; the MLX import is deferred inside
`load_qwen`, so the import here needs only stdlib + numpy. (The sidecar env pins
`mlx-audio`, which has no Linux wheel, so `uv run pytest` is signoff-only.)
"""

import math
import os

import pytest

from qwen3_tts_sidecar import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    handle_synth,
    resolve_gen_params,
)


class _FakeResult:
    audio = [0.0, 0.0]
    sample_rate = 24000


class _FakeModel:
    """Captures generate_custom_voice kwargs so language handling is assertable
    without MLX (the real model import stays deferred inside load_qwen)."""

    def __init__(self):
        self.calls = []

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult()]


def test_handle_synth_defaults_language_to_auto():
    model = _FakeModel()
    path = handle_synth(model, {"dylan": "dylan"}, {"text": "hi", "speaker": "dylan"})
    assert model.calls[0]["language"] == "auto"
    os.remove(path)


def test_handle_synth_reads_language_from_request():
    model = _FakeModel()
    req = {"text": "hallo", "speaker": "dylan", "language": "german"}
    path = handle_synth(model, {"dylan": "dylan"}, req)
    assert model.calls[0]["language"] == "german"
    os.remove(path)


def test_absent_params_use_defaults():
    assert resolve_gen_params({}) == (DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS)


def test_present_valid_params_pass_through():
    assert resolve_gen_params({"temperature": 1.2, "max_tokens": 512}) == (1.2, 512)


def test_temperature_boundaries():
    # (0, 2] — the open lower / closed upper bounds.
    assert resolve_gen_params({"temperature": 2.0})[0] == 2.0
    assert resolve_gen_params({"temperature": 0.0})[0] == DEFAULT_TEMPERATURE
    assert resolve_gen_params({"temperature": 2.1})[0] == DEFAULT_TEMPERATURE
    assert resolve_gen_params({"temperature": -0.5})[0] == DEFAULT_TEMPERATURE


def test_temperature_integer_is_accepted():
    assert resolve_gen_params({"temperature": 1})[0] == 1.0


def test_bool_is_rejected_symmetrically():
    # Both params must reject it (the "why" lives in resolve_gen_params' docstring).
    assert resolve_gen_params({"temperature": True}) == (DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS)
    assert resolve_gen_params({"max_tokens": True}) == (DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS)


def test_nan_and_inf_temperature_rejected():
    assert resolve_gen_params({"temperature": math.nan})[0] == DEFAULT_TEMPERATURE
    assert resolve_gen_params({"temperature": math.inf})[0] == DEFAULT_TEMPERATURE


@pytest.mark.parametrize("bad", ["1.0", None, [], {}])
def test_non_numeric_temperature_rejected(bad):
    assert resolve_gen_params({"temperature": bad})[0] == DEFAULT_TEMPERATURE


@pytest.mark.parametrize("bad", [0, -1, 4096.0, "512", None])
def test_invalid_max_tokens_rejected(bad):
    assert resolve_gen_params({"max_tokens": bad})[1] == DEFAULT_MAX_TOKENS
