#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import avioflow
from vad_filter_onnx import AutoVadModel, VadConfig

SAMPLE_RATE = 16000
CHUNK_TIME = 0.02
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_TIME)


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
        default=0.4
    )
    return parser.parse_args()

def create_vad_config(threshold):
    config = VadConfig()
    config.sample_rate = SAMPLE_RATE
    config.threshold = threshold
    return config

def avioflow_load_audio(path):
    meta, data = avioflow.load(path, output_sample_rate=SAMPLE_RATE)
    print(meta)
    return data[0]

def test(args):
    config = create_vad_config(args.threshold)
    # Create model (using new constructor API)
    model_handle = AutoVadModel(args.model_path, num_threads=1, device_id=-1)
    print("Model handle created successfully")

    # Initialize instance
    model = model_handle.init(config)
    print("Model instance initialized successfully")
    data = avioflow_load_audio(args.audio_path)
    
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i:i + CHUNK_SIZE]
        segs = model.decode(chunk)
        for s in segs:
            print(f"Chunk Segments: {s}")

    last_seg = model.flush()
    if last_seg.idx != -1:
        print(f"Final segment: {last_seg}")

if __name__ == "__main__":
    args = get_args()
    test(args)
