#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from vad_filter_onnx import AutoVadModel, VadConfig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    return parser.parse_args()


def assert_close(actual, expected):
    assert abs(actual - expected) < 1e-6, f"expected {expected}, got {actual}"


def test_config_api(model_path):
    model_handle = AutoVadModel(model_path, num_threads=1, device_id=-1)

    try:
        model_handle.get_config()
        raise AssertionError("get_config should fail before init")
    except Exception as ex:
        assert "initialized model instance" in str(ex)

    config = VadConfig()
    config.sample_rate = 16000
    config.threshold = 0.45
    model = model_handle.init(config)

    current = model.get_config()
    assert current.sample_rate == 16000
    assert_close(current.threshold, 0.45)

    updated = model.get_config()
    updated.threshold = 0.65
    updated.speech_window_size_ms = 200
    model.setup_config(updated)

    current = model.get_config()
    assert_close(current.threshold, 0.65)
    assert current.speech_window_size_ms == 200

    bad_config = model.get_config()
    bad_config.sample_rate = 8000
    try:
        model.setup_config(bad_config)
        raise AssertionError("setup_config should reject sample_rate changes")
    except Exception as ex:
        assert "sample_rate" in str(ex)


if __name__ == "__main__":
    test_config_api(get_args().model_path)
