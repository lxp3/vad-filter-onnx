#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export causal FireRedVAD as an offline waveform ONNX model.

Input is normalized float waveform speech [batch, num_samples]. The graph
applies PCM scaling, 80-bin log fbank and CMVN, then runs DFSMN with caches
disabled. Output is probs [batch, num_frames, 1] only.
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_onnx_firered_vad import (  # noqa: E402
    FireRedWaveformFrontend,
    add_metadata_to_onnx,
    opset_version,
    simplify_onnx,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Export FireRedVAD Stream-VAD to offline ONNX without caches."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/FireRedTeam/FireRedVAD/Stream-VAD",
        help="Path to FireRedVAD Stream-VAD model directory.",
    )
    default_onnx = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "public",
            "models",
            "firered_vad.offline.onnx",
        )
    )
    parser.add_argument("--onnx-path", type=str, default=default_onnx)
    parser.add_argument("--opset", type=int, default=opset_version)
    parser.add_argument("--skip-simplify", action="store_true")
    parser.add_argument("--verify", type=int, default=1)
    return parser.parse_args()


def load_detect_model():
    try:
        from fireredvad.core.detect_model import DetectModel
    except ImportError as exc:
        try:
            from core.detect_model import DetectModel
        except ImportError:
            raise ImportError(
                "Could not import FireRedVAD DetectModel. Set PYTHONPATH to either "
                "the package root or its fireredvad directory."
            ) from exc
    return DetectModel


class FireRedVadOffline(nn.Module):
    def __init__(self, model, frontend):
        super().__init__()
        self.model = model
        self.frontend = frontend

    def forward(self, speech):
        feat = self.frontend(speech)
        probs, _new_caches = self.model(feat, caches=None)
        return probs


def assert_causal(model):
    blocks = [model.dfsmn.fsmn1]
    blocks.extend(block.fsmn for block in model.dfsmn.fsmns)
    for index, fsmn in enumerate(blocks):
        if int(fsmn.N2) != 0:
            raise RuntimeError(
                f"offline export requires causal FSMN, block {index} has N2={fsmn.N2}"
            )


def verify_onnx(onnx_path, wrapper, speech):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    input_names = [item.name for item in onnx_model.graph.input]
    output_names = [item.name for item in onnx_model.graph.output]
    if input_names != ["speech"] or output_names != ["probs"]:
        raise RuntimeError(f"unexpected IO names inputs={input_names} outputs={output_names}")
    print("ONNX model verified OK")

    with torch.no_grad():
        torch_probs = wrapper(speech)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_probs = session.run(["probs"], {"speech": speech.cpu().numpy()})[0]
    prob_diff = float(np.max(np.abs(ort_probs - torch_probs.cpu().numpy())))
    print(f"PyTorch vs ONNX max abs diff: probs={prob_diff:.8f}")
    if prob_diff >= 1e-4:
        raise RuntimeError(f"probs diff too large: {prob_diff}")


def export_onnx(model_dir, output_path, opset, skip_simplify, verify):
    detect_model = load_detect_model()
    model_path = os.path.join(model_dir, "model.pth.tar")
    cmvn_path = os.path.join(model_dir, "cmvn.ark")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.isfile(cmvn_path):
        raise FileNotFoundError(f"Missing CMVN file: {cmvn_path}")

    print(f"Loading FireRedVAD Stream-VAD from: {model_dir}")
    model = detect_model.from_pretrained(model_dir).cpu().eval()
    assert_causal(model)
    frontend = FireRedWaveformFrontend(cmvn_path)
    wrapper = FireRedVadOffline(model, frontend).cpu().eval()

    dummy_speech = torch.randn(1, 16000, dtype=torch.float32).clamp(-1.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_speech,),
        output_path,
        input_names=["speech"],
        output_names=["probs"],
        dynamic_axes={
            "speech": {0: "batch", 1: "num_samples"},
            "probs": {0: "batch", 1: "num_frames"},
        },
        opset_version=opset,
        verbose=False,
        dynamo=False,
    )
    print(f"Exported offline model to: {output_path}")

    if not skip_simplify:
        simplify_onnx(output_path)

    metadata = {
        "model_type": "firered_vad",
        "sample_rate": 16000,
        "input_scale": "normalized_float",
        "streaming": 0,
    }
    add_metadata_to_onnx(output_path, metadata)
    size = os.path.getsize(output_path)
    print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
    if verify:
        verify_onnx(output_path, wrapper, dummy_speech)


def main():
    args = get_args()
    export_onnx(
        model_dir=args.model_dir,
        output_path=args.onnx_path,
        opset=args.opset,
        skip_simplify=args.skip_simplify,
        verify=bool(args.verify),
    )


if __name__ == "__main__":
    sys.exit(main())
