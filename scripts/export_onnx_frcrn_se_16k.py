#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxsim import simplify

opset_version = 18


"""
FRCRN offline denoise export notes
===================================

FRCRN (speech_frcrn_ans_cirm_16k) is a fully offline/non-causal denoise
model: whole-utterance ConvSTFT/ConviSTFT, non-causal U-Net convolutions,
and global-time-average SE gating with no recurrent state at all. Unlike
this project's other denoise models (GTCRN, DPDFNet), it cannot be run
incrementally chunk-by-chunk and carries no cache/state tensors across
calls.

FRCRN's own forward() already takes the raw waveform directly and performs
STFT/iSTFT internally via ConvSTFT/ConviSTFT (nn.Conv1d/ConvTranspose1d
against a fixed non-trainable DFT kernel, not torch.stft), so this exporter
needs no hand-rolled frontend module: the entire feature-extraction
pipeline is already baked into the model graph.

Audio format
------------
- Sample rate: 16000 Hz only.
- Input tensor name: speech.
- Input tensor shape: [1, num_samples].
- Input dtype: float32, normalized float in [-1, 1].
- Output tensor name: enhanced.
- Output tensor shape: [1, num_samples'] float32 (see length note below).

Model internals
----------------
- STFT: win_len=640 (40 ms), win_inc=320 (20 ms hop), fft_len=640, hann
  window, complex feature type.
- forward() returns a 6-tensor list:
  [est_spec1, est_wav1, est_mask1, est_spec2, est_wav2, est_mask2].
  The final enhanced waveform is out_list[4] (the deeper/second-stage
  refinement, matching modelscope's own pipeline "output_pcm"/wav_l2).
- No recurrent cache: UNet/UNet2, ComplexUniDeepFsmn(_L1), and SELayer are
  all feedforward given the whole utterance; there is no caches_in/out I/O.

Length note
-----------
- ConvSTFT does not center-pad, so ConviSTFT's overlap-add reconstruction
  produces an output length that is not exactly the input length (edge
  effect of the analysis window). This is inherent to the upstream model
  and is not corrected here; verify() checks PyTorch-vs-ONNX agreement on
  whatever length the model itself produces, at more than one input
  length, to also confirm the traced graph's dynamic-time-axis reshapes
  (inside ComplexUniDeepFsmn/SELayer, which read x.size() at trace time)
  generalize correctly.

Streaming
---------
- streaming=0 in ONNX metadata. This model must never be called
  incrementally; the C++ backend (FrcrnSe16kDenoiseModel) buffers all input and
  only invokes the ONNX session once, on input_finished=true.
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="Export FRCRN offline denoise model to ONNX with STFT/iSTFT baked in."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/user/lxp/llm/downloads/models/iic/speech_frcrn_ans_cirm_16k",
        help="Path to the FRCRN modelscope checkpoint directory.",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="public/models/frcrn_se_16k.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=opset_version,
        help=f"ONNX opset version. Default: {opset_version}.",
    )
    parser.add_argument(
        "--skip-simplify",
        action="store_true",
        help="Skip onnxsim simplification.",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=1,
        help="Run PyTorch vs ONNX Runtime verification after export.",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        default=1,
        help="Save dynamic int8 model next to ONNX using .onnx -> .int8.onnx.",
    )
    return parser.parse_args()


def load_frcrn_se_16k(model_dir):
    from modelscope.models.audio.ans.frcrn import FRCRN

    with open(os.path.join(model_dir, "configuration.json")) as f:
        cfg = json.load(f)["model"]
    cfg = dict(cfg)
    cfg.pop("type", None)

    model = FRCRN(**cfg)
    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu"
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={missing} unexpected={unexpected}")
    return model.eval()


class FrcrnSe16kOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, speech):
        out_list = self.model(speech)
        return out_list[4]


def add_metadata_to_onnx(onnx_path, metadata_dict):
    model = onnx.load(onnx_path)
    del model.metadata_props[:]
    for key, value in metadata_dict.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, onnx_path)
    print(f"Added metadata: {metadata_dict}")


def simplify_onnx(onnx_path):
    model = onnx.load(onnx_path)
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, onnx_path)
    print("Simplified with onnxsim")


def dedupe_shared_gemm_weights(input_path, output_path):
    """
    FRCRN's SE layers apply the same fc_r/fc_i Linear module to two separate
    tensors, so the exported graph has multiple Gemm nodes reading the same
    weight initializer. ONNX Runtime's dynamic quantizer unconditionally
    rewrites every Gemm into MatMul(+Add) and, for transB=1, transposes the
    weight initializer in place; when two Gemm nodes share that initializer,
    the second one gets transposed twice, producing an incompatible shape
    and a ShapeInferenceError at load time. Give each Gemm node's weight
    input its own private copy of the initializer to sidestep the bug. This
    is only applied to a throwaway copy feeding the int8 pipeline; the
    float ONNX file at input_path is left untouched.
    """
    model = onnx.load(input_path)
    usage_count = {}
    for node in model.graph.node:
        if node.op_type == "Gemm" and len(node.input) > 1:
            usage_count[node.input[1]] = usage_count.get(node.input[1], 0) + 1

    initializers_by_name = {init.name: init for init in model.graph.initializer}
    seen_once = set()
    duplicated = 0
    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) <= 1:
            continue
        weight_name = node.input[1]
        if usage_count.get(weight_name, 0) <= 1:
            continue
        if weight_name not in seen_once:
            seen_once.add(weight_name)
            continue
        original = initializers_by_name[weight_name]
        new_init = onnx.TensorProto()
        new_init.CopyFrom(original)
        new_init.name = f"{weight_name}__dup{duplicated}"
        duplicated += 1
        model.graph.initializer.append(new_init)
        node.input[1] = new_init.name

    print(f"Duplicated {duplicated} shared Gemm weight initializers")
    onnx.save(model, output_path)


def quantize_onnx_model(input_path, output_path):
    from onnxruntime.quantization.shape_inference import quant_pre_process

    deduped_path = input_path.replace(".onnx", ".deduped.onnx")
    dedupe_shared_gemm_weights(input_path, deduped_path)

    preprocessed_path = input_path.replace(".onnx", ".preproc.onnx")
    quant_pre_process(deduped_path, preprocessed_path, skip_symbolic_shape=True)
    os.remove(deduped_path)

    # Node names must be read from the preprocessed graph, not the original:
    # quant_pre_process's shape-inference/optimization pass can rename nodes
    # (e.g. Gemm folding), so exclusions computed against the original graph
    # silently fail to match and previously-excluded ops got quantized anyway.
    model = onnx.load(preprocessed_path)
    nodes_to_exclude = []
    preprocess_inits = []
    preprocess_keywords = (
        "stft",
        "istft",
        "window",
        "dft",
        "se_layer",
    )

    for init in model.graph.initializer:
        if any(keyword in init.name.lower() for keyword in preprocess_keywords):
            preprocess_inits.append(init.name)

    for node in model.graph.node:
        node_name = node.name.lower()
        if any(inp in preprocess_inits for inp in node.input):
            nodes_to_exclude.append(node.name)
            continue
        if any(keyword in node_name for keyword in preprocess_keywords):
            nodes_to_exclude.append(node.name)
            continue
        if node.op_type == "Conv":
            group = next((a.i for a in node.attribute if a.name == "group"), 1)
            if group != 1:
                # ONNX Runtime's CPU ConvInteger kernel does not support
                # grouped/depthwise convolution (UniDeepFsmn.conv1 uses
                # groups=output_dim), so quantizing these produces a model
                # that fails to load at inference time.
                nodes_to_exclude.append(node.name)

    nodes_to_exclude = sorted(set(nodes_to_exclude))
    print(f"Excluding {len(nodes_to_exclude)} nodes from int8 quantization")

    # Gemm nodes are internally decomposed into MatMul+Add by the quantizer
    # before exclusion is applied, so a Gemm node's original name never
    # matches the synthesized MatMul node it produces and nodes_to_exclude
    # silently fails to protect it (observed as a ShapeInferenceError on
    # SE-layer Gemms at load time). Restrict quantization to Conv only,
    # which is unambiguous and covers the vast majority of this model's
    # weights; Gemm/MatMul (FSMN projections, SE-layer FC) stay float32.
    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=["Conv"],
        per_channel=False,
        reduce_range=False,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    os.remove(preprocessed_path)
    print(f"Quantized int8 model saved to: {output_path}")


def verify_onnx(onnx_path, wrapper, num_samples_list):
    import numpy as np
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified OK")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    for num_samples in num_samples_list:
        speech = torch.randn(1, num_samples, dtype=torch.float32).clamp(-1.0, 1.0)
        with torch.no_grad():
            torch_out = wrapper(speech)

        ort_out = session.run(None, {"speech": speech.cpu().numpy()})[0]

        length = min(torch_out.shape[1], ort_out.shape[1])
        diff = float(
            np.max(np.abs(ort_out[:, :length] - torch_out.cpu().numpy()[:, :length]))
        )
        print(
            f"num_samples={num_samples}: torch_shape={tuple(torch_out.shape)} "
            f"onnx_shape={tuple(ort_out.shape)} max_abs_diff={diff:.8f}"
        )
        assert diff < 1e-4, f"enhanced diff too large at num_samples={num_samples}: {diff}"


def export_onnx(model_dir, output_path, opset, skip_simplify, verify, quantize):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    print(f"Loading FRCRN from: {model_dir}")
    model = load_frcrn_se_16k(model_dir)
    wrapper = FrcrnSe16kOnnxWrapper(model).eval()

    dummy_speech = torch.randn(1, 16000, dtype=torch.float32).clamp(-1.0, 1.0)
    dummy_inputs = (dummy_speech,)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=["speech"],
        output_names=["enhanced"],
        dynamic_axes={
            "speech": {1: "num_samples"},
            "enhanced": {1: "num_samples"},
        },
        opset_version=opset,
        verbose=False,
        dynamo=False,
    )
    print(f"Exported model to: {output_path}")

    if not skip_simplify:
        simplify_onnx(output_path)

    metadata = {
        "model_type": "frcrn_se_16k_denoise",
        "sample_rate": 16000,
        "input_scale": "normalized_float",
        "streaming": 0,
    }
    add_metadata_to_onnx(output_path, metadata)

    size = os.path.getsize(output_path)
    print(f"File size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

    if verify:
        verify_onnx(output_path, wrapper, [16000, 24001])

    if quantize:
        quantized_path = output_path.replace(".onnx", ".int8.onnx")
        if quantized_path == output_path:
            raise ValueError(f"ONNX path must end with .onnx for int8 output: {output_path}")
        quantize_onnx_model(output_path, quantized_path)
        add_metadata_to_onnx(quantized_path, metadata)
        quantized_size = os.path.getsize(quantized_path)
        print(
            f"Int8 file size: {quantized_size:,} bytes "
            f"({quantized_size / 1024 / 1024:.2f} MB)"
        )


def main():
    args = get_args()
    export_onnx(
        model_dir=args.model_dir,
        output_path=args.onnx_path,
        opset=args.opset,
        skip_simplify=args.skip_simplify,
        verify=bool(args.verify),
        quantize=bool(args.quantize),
    )


if __name__ == "__main__":
    sys.exit(main())
