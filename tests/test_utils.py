from pathlib import Path

import pytest

from casc_lite.core.utils import ExperimentConfig, extract_number, load_config, normalize_text


def test_load_config_applies_overrides(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("a: 0.5\nmode: fixed\nn_fixed: 3\n")
    config = load_config(config_path, overrides={"top_p": 0.8, "n_candidates": "1,3,5"})
    assert isinstance(config, ExperimentConfig)
    assert config.a == pytest.approx(0.5)
    assert config.mode == "fixed"
    assert config.n_fixed == 3
    assert config.top_p == pytest.approx(0.8)
    assert config.n_candidates == [1, 3, 5]


def test_extract_number_returns_last_match():
    text = "First 3 then 7.5 at the end 42"
    assert extract_number(text) == "42"
    assert extract_number("no digits") is None


def test_normalize_text_lowercases_and_strips():
    assert normalize_text("  Hello\nWorld  ") == "hello world"
