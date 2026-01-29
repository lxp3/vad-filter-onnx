import os
import sys

# Update sys.path to support both build and build_static (Windows and Linux)
possible_paths = [
    "../../build/vad-filter-onnx/python/Release",
    "../../build_static/vad-filter-onnx/python/Release",
    "../../build_shared/vad-filter-onnx/python/Release",
    "../../build/vad-filter-onnx/python",
    "../../build_static/vad-filter-onnx/python",
    "../../build_shared/vad-filter-onnx/python",
]

found_path = False
for p in possible_paths:
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), p))
    if os.path.exists(full_path):
        print(f"Adding library path: {full_path}")
        sys.path.append(full_path)
        found_path = True
        break

if not found_path:
    print("Warning: Could not find any compiled library paths. Checked:")
    for p in possible_paths:
        print(f"  - {os.path.abspath(os.path.join(os.path.dirname(__file__), p))}")

try:
    from avioflow import AudioDecoder, AudioStreamOptions
    from vad_filter_onnx import get_ort_available_providers, AutoVadModel, VadConfig
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def test_online_vad():
    print("Available providers:", get_ort_available_providers())

    # Create config
    config = VadConfig()
    config.sample_rate = 16000
    config.threshold = 0.5
    print(f"Config initialized: sample_rate={config.sample_rate}, threshold={config.threshold}")

    # Paths
    model_path = os.path.join(os.path.dirname(__file__), "../../public/models/fsmn_vad.16k.onnx")
    audio_path = os.path.join(os.path.dirname(__file__), "../../public/wavs/zh.wav")
    
    # Load and resample audio
    print(f"Loading audio from {audio_path}")
    # Using avioflow to read audio
    options = AudioStreamOptions()
    options.output_sample_rate = 16000
    options.output_num_channels = 1
    decoder = AudioDecoder(options)
    decoder.open(audio_path)
    metadata = decoder.get_metadata()
    print(f"\nMetadata Recognized:")
    print(f"  Container:    {metadata.container}")
    print(f"  Codec:        {metadata.codec}")
    print(f"  Sample Rate:  {metadata.sample_rate} Hz")
    print(f"  Channels:     {metadata.num_channels}")
    print(f"  Duration:     {metadata.duration:.3f} s")
    print(f"  Num Samples:  {metadata.num_samples}")
    
    # Get all samples (returns list of numpy arrays, one per channel)
    samples = decoder.get_all_samples()
    # Assume mono or take first channel
    mono_data = samples.data[0]
    
    # Create model (using new constructor API)
    model_handle = AutoVadModel(model_path, num_threads=1, device_id=-1)
    print("Model handle created successfully")

    # Initialize instance
    instance = model_handle.init(config)
    print("Model instance initialized successfully")

    # Decode in 100ms chunks to simulate online behavior
    sample_rate = config.sample_rate
    chunk_size = int(sample_rate * 0.1) # 100ms
    
    print(f"Decoding in chunks of {chunk_size} samples (100ms)...")
    
    all_segments = []
    for i in range(0, len(mono_data), chunk_size):
        chunk = mono_data[i:i + chunk_size]
        segments = instance.decode(chunk)
        if segments:
            all_segments.extend(segments)
            for s in segments:
                print(f"Chunk Segments: {s}")

    # Finalize
    print("\nFlushing...")
    last_segment = instance.flush()
    if last_segment.idx != -1:
        print(f"Final segment: {last_segment}")
        all_segments.append(last_segment)

    print(f"\nAll Decoded segments: {all_segments}")

if __name__ == "__main__":
    test_online_vad()
