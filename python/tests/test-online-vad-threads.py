#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import avioflow
import soundfile as sf
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
        "--wavscp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vadscp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
    )
    return parser.parse_args()


def create_vad_config(threshold):
    config = VadConfig()
    config.sample_rate = SAMPLE_RATE
    config.threshold = threshold
    return config


def avioflow_load_audio(path):
    _, data = avioflow.load(path, output_sample_rate=SAMPLE_RATE)
    return data[0]


def load_wav_scp(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, wav_path = line.split(maxsplit=1)
            key = key.replace(".wav", "")
            items.append((key, wav_path))
    return items

def save_audio(save_dir, key, wav_path, start_ms, end_ms):
    if not save_dir:
        return

    os.makedirs(save_dir, exist_ok=True)

    meta, data = avioflow.load(wav_path)
    sample_rate = meta.sample_rate
    start = int(sample_rate * 0.001 * start_ms)
    end = int(sample_rate * 0.001 * end_ms)
    seg_data = data[0][start:end]
    savepath = f"{save_dir}/{key}.wav"
    sf.write(savepath, seg_data, sample_rate)


def decode_one(model_handle, config, item, save_dir):
    key, wav_path = item
    model = model_handle.init(config)
    data = avioflow_load_audio(wav_path)
    segments = []

    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i:i + CHUNK_SIZE]
        segs = model.decode(chunk)
        for seg in segs:
            if seg.end_ms > 0:
                seg_key = f"{key}-{seg.idx}"
                segments.append((seg_key, seg))
                save_audio(save_dir=save_dir, key=seg_key, wav_path=wav_path, start_ms=seg.start_ms, end_ms=seg.end_ms)

    last_seg = model.flush()
    if last_seg.idx != -1 and last_seg.end_ms > 0:
        seg_key = f"{key}-{last_seg.idx}"
        segments.append((seg_key, last_seg))
        save_audio(save_dir=save_dir, key=seg_key, wav_path=wav_path, start_ms=last_seg.start_ms, end_ms=last_seg.end_ms)
    return segments


def test(args):
    config = create_vad_config(args.threshold)
    wavscp = load_wav_scp(args.wavscp)
    model_handle = AutoVadModel(args.model_path, num_threads=1, device_id=-1)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor, open(args.vadscp, "w", encoding="utf-8") as f:
        results = executor.map(
            lambda item: decode_one(model_handle, config, item, args.save_dir),
            wavscp,
        )

        for segments in results:
            for key, seg in segments:
                print(f"{key} {(seg.start_ms * 0.001):.3f}  {(seg.end_ms * 0.001):.3f}", file=f)


if __name__ == "__main__":
    args = get_args()
    test(args)
