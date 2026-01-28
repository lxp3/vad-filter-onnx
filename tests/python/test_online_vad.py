import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../build/vad-filter-onnx/python/Release"))
from avioflow import AudioDecoder, AudioStreamOptions
import vad_filter_onnx as vad

def test_online_vad():
    print("Available providers:", vad.get_ort_available_providers())

    # Create config
    config = vad.VadConfig()
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
    
    # Create model
    model_handle = vad.AutoVadModel.create(model_path, num_threads=1, device_id=-1)
    print("Model handle created successfully")

    # Initialize instance
    instance = model_handle.init(config)
    print("Model instance initialized successfully")

    # Decode in chunks to simulate online behavior if desired, or all at once
    # For simplicity, decode all at once
    segments = instance.decode(mono_data, True)
    print(f"Decoded segments: {segments}")

    for segment in segments:
        print(f"Segment: {segment}")

    # Finalize
    last_segment = instance.flush()
    if last_segment.idx != -1:
        print(f"Final segment: {last_segment}")

if __name__ == "__main__":
    test_online_vad()
