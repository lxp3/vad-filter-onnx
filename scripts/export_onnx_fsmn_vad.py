#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
from funasr import AutoModel
from funasr.models.fsmn_vad_streaming.encoder import BasicBlock, BasicBlock_export
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxsim import simplify
from filter_fbank import Filterbank

opset_version = 18


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=str, default="models/iic/speech_fsmn_vad_zh-cn-8k-common"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=8000, help="Sample rate (8000 or 16000)"
    )
    parser.add_argument(
        "--quantize", type=int, default=1, help="Export quantized int8 model"
    )
    return parser.parse_args()


class StreamingFbankLFR(nn.Module):
    def __init__(self, sample_rate, cmvn, feat_dim=80, lfr_m=5):
        super().__init__()
        assert lfr_m == 5, "当前实现假设 lfr_m = 5"
        self.cmvn = cmvn
        self.feat_dim = feat_dim
        self.lfr_m = lfr_m
        self.left_ctx = lfr_m // 2  # 2

        self.fbank = Filterbank(
            sample_rate=sample_rate,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            window_fn=torch.hamming_window,
            snip_edges=True,
            onnx_compatible=True,
        )

    def compute_feats(
        self,
        waveforms: torch.Tensor,
        first_padding: torch.Tensor,
        last_padding: torch.Tensor,
    ):
        """
        Args:
            waveforms: (B, T)
        Returns:
            feats: (B, T_padded, 80)
        """
        assert first_padding.dtype == torch.int64 and last_padding.dtype == torch.int64
        waveforms = waveforms * (1 << 15)
        feats = self.fbank(waveforms)
        feats = torch.nn.functional.pad(
            feats,
            (
                0,
                0,
                first_padding,
                last_padding,
            ),
            mode="replicate",
        )

        return feats

    def apply_lfr(self, feats: torch.Tensor):
        """
        Args:
            feats: (B, T, 80)

        Returns:
            y:         (B, T-4, 5*80)
        """
        T_in = feats.size(1)
        T_out = T_in - 4

        y = torch.cat(
            [
                feats[:, 0:T_out, :],
                feats[:, 1 : T_out + 1, :],
                feats[:, 2 : T_out + 2, :],
                feats[:, 3 : T_out + 3, :],
                feats[:, 4 : T_out + 4, :],
            ],
            dim=-1,
        )  # (B, T_out, 400)

        return y

    def apply_cmvn(self, lfr_feats: torch.Tensor):
        """
        Args:
            lfr_feats (torch.Tensor): Input LFR features tensor (B, T, 400)
        """
        lfr_feats = self.cmvn[0:1, :] + lfr_feats
        lfr_feats = self.cmvn[1:2, :] * lfr_feats
        return lfr_feats

    def forward(
        self,
        waveforms: torch.Tensor,
        first_padding: torch.Tensor,
        last_padding: torch.Tensor,
    ):
        """
        Args:
            waveforms (torch.Tensor): Input waveforms tensor (B, T).
            first_padding (torch.Tensor): Number of frames to pad at start.
            last_padding (torch.Tensor): Number of frames to pad at end.
        Returns:
            lfr_feats (torch.Tensor): Output LFR features tensor
        """
        feats = self.compute_feats(
            waveforms, first_padding=first_padding, last_padding=last_padding
        )

        lfr_feats = self.apply_lfr(feats)
        lfr_feats = self.apply_cmvn(lfr_feats)
        return lfr_feats


class FsmnVadStreamingExport(nn.Module):
    def __init__(self, model: nn.Module, preprocess_module: StreamingFbankLFR):
        super().__init__()
        self.preprocess_module = preprocess_module
        self.in_linear1 = model.in_linear1
        self.in_linear2 = model.in_linear2
        self.relu = model.relu
        self.out_linear1 = model.out_linear1
        self.out_linear2 = model.out_linear2
        self.softmax = model.softmax
        self.fsmn = model.fsmn
        for i, d in enumerate(model.fsmn):
            if isinstance(d, BasicBlock):
                self.fsmn[i] = BasicBlock_export(d)

    def forward(
        self,
        waveforms: torch.Tensor,
        in_cache0: torch.Tensor,
        in_cache1: torch.Tensor,
        in_cache2: torch.Tensor,
        in_cache3: torch.Tensor,
        first_padding: torch.Tensor,
        last_padding: torch.Tensor,
    ):
        """
        Args:
            waveforms (torch.Tensor): Input waveforms tensor (B, T)
            in_cache0, in_cache1, in_cache2, in_cache3: Input cache tensors
            first_padding (torch.Tensor): Frames to pad at start
            last_padding (torch.Tensor): Frames to pad at end
        """
        # LFR 特征提取
        x = self.preprocess_module(
            waveforms, first_padding=first_padding, last_padding=last_padding
        )

        # 线性层
        x = self.in_linear1(x)
        x = self.in_linear2(x)
        x = self.relu(x)

        # FSMN 层
        caches = [in_cache0, in_cache1, in_cache2, in_cache3]
        out_caches = []
        for i, d in enumerate(self.fsmn):
            x, out_cache = d(x, caches[i])
            out_caches.append(out_cache)

        # 输出层
        x = self.out_linear1(x)
        x = self.out_linear2(x)
        x = self.softmax(x)

        # [B, T, 248] -> [B, T]
        x = x[:, :, 0]

        return (
            x,
            out_caches[0],
            out_caches[1],
            out_caches[2],
            out_caches[3],
        )


def create_dummy_inputs(encoder_conf):
    """创建用于ONNX导出的dummy inputs"""
    num_samples = 1 * 16000
    lorder = encoder_conf.get("lorder")
    rorder = encoder_conf.get("rorder")
    cache_frames = lorder + rorder - 1
    proj_dim = encoder_conf.get("proj_dim")

    waveforms = torch.randn(1, num_samples, dtype=torch.float32)
    in_cache0 = torch.randn(1, proj_dim, cache_frames, 1, dtype=torch.float32)
    in_cache1 = torch.randn(1, proj_dim, cache_frames, 1, dtype=torch.float32)
    in_cache2 = torch.randn(1, proj_dim, cache_frames, 1, dtype=torch.float32)
    in_cache3 = torch.randn(1, proj_dim, cache_frames, 1, dtype=torch.float32)
    first_padding = torch.tensor(0, dtype=torch.int64)
    last_padding = torch.tensor(0, dtype=torch.int64)

    return (
        waveforms,
        in_cache0,
        in_cache1,
        in_cache2,
        in_cache3,
        first_padding,
        last_padding,
    )


def add_metadata_to_onnx(onnx_path, metadata_dict):
    """给ONNX模型添加自定义metadata"""
    model = onnx.load(onnx_path)

    # simplify model
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    # 添加metadata
    for key, value in metadata_dict.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    # 保存模型
    onnx.save(model, onnx_path)
    print(f"Added metadata: {metadata_dict}")


def quantize_onnx_model(input_path, output_path):
    """量化ONNX模型为int8，排除预处理部分（StreamingFbankLFR）以保持精度"""
    # 加载模型并找到需要排除的节点
    model = onnx.load(input_path)
    nodes_to_exclude = []

    # 1. 查找预处理相关的初始化值名称 (Filterbank 权重, CMVN 等)
    # PyTorch 导出的名称通常包含模块名，如 "preprocess_module.fbank.mel_filters"
    preprocess_inits = []
    for init in model.graph.initializer:
        if "preprocess_module" in init.name or "fbank" in init.name:
            preprocess_inits.append(init.name)

    # 2. 遍历节点，识别属于预处理部分的计算节点
    for node in model.graph.node:
        # 排除所有卷积层 (避免 ConvInteger 兼容性问题)
        if node.op_type == "Conv":
            nodes_to_exclude.append(node.name)
            print(f"Excluding node: {node.name}")
            continue
        # 如果节点使用了预处理部分的权重 (如 Fbank 的 MatMul)，则排除该节点
        # 这样可以确保这些操作保持 float32 计算
        if any(inp in preprocess_inits for inp in node.input):
            nodes_to_exclude.append(node.name)
            print(f"Excluding node: {node.name}")
            continue

        # 另外排除带有 preprocess 关键词的计算类节点
        if "preprocess" in node.name.lower() and node.op_type in [
            "MatMul",
            "Gemm",
            "Gather",
        ]:
            nodes_to_exclude.append(node.name)
            print(f"Excluding node: {node.name}")

    # 去重
    nodes_to_exclude = list(set(nodes_to_exclude))
    print(f"Excluding {len(nodes_to_exclude)} nodes from quantization (Pre-processing)")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=False,
        reduce_range=False,
    )
    print(f"Quantized model saved to: {output_path}")


def export_onnx(model_dir, output_path, sample_rate=8000, quantize=False):
    """导出FsmnVadStreaming模型为ONNX格式"""
    # 加载模型
    model = AutoModel(model=model_dir, device="cpu", disable_update=True)
    cmvn: torch.Tensor = model.kwargs["frontend"].cmvn.to("cpu")
    encoder_conf = model.model.encoder_conf
    print(f"encoder_conf: {encoder_conf}")

    # 创建导出模型
    print(model.model.encoder)
    preprocess_module = StreamingFbankLFR(sample_rate, cmvn)
    export_model = FsmnVadStreamingExport(model.model.encoder, preprocess_module).to(
        "cpu"
    )

    # 创建dummy inputs
    dummy_inputs = create_dummy_inputs(encoder_conf)

    # 设置模型为评估模式
    export_model.eval()

    # 导出为ONNX (float32)
    torch.onnx.export(
        export_model,
        dummy_inputs,
        output_path,
        input_names=[
            "speech",
            "in_cache0",
            "in_cache1",
            "in_cache2",
            "in_cache3",
            "first_padding",
            "last_padding",
        ],
        output_names=[
            "logits",
            "out_cache0",
            "out_cache1",
            "out_cache2",
            "out_cache3",
        ],
        dynamic_axes={
            "speech": {1: "num_samples"},
            "logits": {1: "num_frames"},
        },
        opset_version=opset_version,
        verbose=False,
        dynamo=False,
    )

    print(f"ONNX model exported to: {output_path}")
    print(f"Note: Model uses slicing for LFR feature extraction")

    # 添加metadata
    metadata = {
        "model_type": "fsmn_vad",
        "sample_rate": sample_rate,
    }
    add_metadata_to_onnx(output_path, metadata)

    # 量化模型
    if quantize:
        quantized_path = output_path.replace(".onnx", ".int8.onnx")
        quantize_onnx_model(output_path, quantized_path)
        # 同样给量化模型添加metadata
        add_metadata_to_onnx(quantized_path, metadata)


if __name__ == "__main__":
    args = get_args()

    # 根据sample_rate确定输出文件名
    if args.sample_rate == 8000:
        onnx_filename = "fsmn_vad.8k.onnx"
    elif args.sample_rate == 16000:
        onnx_filename = "fsmn_vad.16k.onnx"
    else:
        raise ValueError(f"Invalid sample rate: {args.sample_rate}")

    onnx_path = os.path.join(args.model_dir, onnx_filename)
    export_onnx(args.model_dir, onnx_path, args.sample_rate, args.quantize)
