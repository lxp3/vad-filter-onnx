#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import avioflow
from vad_filter_onnx import AutoVadModel, VadConfig

SAMPLE_RATE = 16000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def create_vad_config(threshold):
    config = VadConfig()
    config.sample_rate = SAMPLE_RATE
    config.threshold = threshold
    return config


def avioflow_load_audio(path):
    meta, data = avioflow.load(
        path,
        output_sample_rate=SAMPLE_RATE
    )
    print(meta)
    return data[0]


def test(args):
    config = create_vad_config(args.threshold)
    model_handle = AutoVadModel(args.model_path, num_threads=1, device_id=-1)
    print("Model handle created successfully")

    model = model_handle.init(config)
    print("Model instance initialized successfully")
    data = avioflow_load_audio(args.audio_path)

    segs = model.decode(data, input_finished=True)
    for seg in segs:
        print(f"Segments: {seg}")

    print(segs)


if __name__ == "__main__":
    args = get_args()
    test(args)
