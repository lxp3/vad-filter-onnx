#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from vad_filter_onnx import AutoVadModel, VadConfig


def test_webrtc_config():
    handle = AutoVadModel.create_webrtc()
    config = VadConfig()
    config.sample_rate = 16000
    config.webrtc_vad_mode = 3
    config.webrtc_frame_ms = 30
    model = handle.init(config)

    current = model.get_config()
    assert current.webrtc_vad_mode == 3
    assert current.webrtc_frame_ms == 30

    current.webrtc_vad_mode = 2
    current.webrtc_frame_ms = 10
    model.setup_config(current)
    current = model.get_config()
    assert current.webrtc_vad_mode == 2
    assert current.webrtc_frame_ms == 10

    bad = model.get_config()
    bad.sample_rate = 8000
    try:
        model.setup_config(bad)
        raise AssertionError("setup_config should reject sample_rate changes")
    except Exception as ex:
        assert "sample_rate" in str(ex)


if __name__ == "__main__":
    test_webrtc_config()
    print("ok")
