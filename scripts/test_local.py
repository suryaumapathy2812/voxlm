#!/usr/bin/env python3
"""
Quick local test for VoxLM 0.5B model.

Tests:
1. Model loads correctly
2. Audio encoder works
3. Forward pass works
4. Transcribe function works

Run: python scripts/test_local.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_imports():
    """Test that all imports work."""
    print("=" * 50)
    print("TEST 1: Imports")
    print("=" * 50)

    try:
        from src.config import get_config, MODEL_CONFIGS
        from src.audio_encoder import AudioEncoder, AudioProjection
        from src.heads import TimestampHead, ConfidenceHead
        from src.model import VoxLM

        print("All imports successful")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False


def test_audio_encoder():
    """Test audio encoder independently."""
    print("\n" + "=" * 50)
    print("TEST 2: Audio Encoder")
    print("=" * 50)

    from src.audio_encoder import AudioEncoder, AudioProjection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test with whisper-tiny (smallest)
    print("Loading Whisper-tiny encoder...")
    encoder = AudioEncoder("openai/whisper-tiny", freeze=True)
    encoder = encoder.to(device)
    print(f"✓ Encoder loaded (dim={encoder.hidden_dim})")

    # Create test audio (5 seconds)
    sample_rate = 16000
    duration = 5
    test_audio = torch.randn(sample_rate * duration).to(device)
    print(f"Test audio: {duration}s @ {sample_rate}Hz")

    # Encode
    with torch.no_grad():
        features, mask = encoder(test_audio.unsqueeze(0))

    print(f"✓ Encoded: {features.shape}")
    print(f"  Frames: {features.shape[1]} ({features.shape[1] / 50:.2f}s at 50Hz)")

    # Test projection
    projection = AudioProjection(
        audio_dim=encoder.hidden_dim,
        llm_dim=896,  # Qwen2-0.5B dim
        downsample_factor=2,
    ).to(device)

    with torch.no_grad():
        projected, new_mask = projection(features, mask)

    print(f"✓ Projected: {projected.shape}")
    print(f"  Downsampled to {projected.shape[1]} frames (25Hz)")

    return True


def test_heads():
    """Test output heads."""
    print("\n" + "=" * 50)
    print("TEST 3: Output Heads")
    print("=" * 50)

    from src.heads import TimestampHead, ConfidenceHead

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 896

    # Dummy hidden states
    batch_size = 2
    seq_len = 20
    hidden = torch.randn(batch_size, seq_len, hidden_dim).to(device)

    # Timestamp head
    ts_head = TimestampHead(hidden_dim, max_frames=1500).to(device)
    start_logits, end_logits = ts_head(hidden, num_audio_frames=100)
    print(f"✓ TimestampHead: start={start_logits.shape}, end={end_logits.shape}")

    # Decode timestamps
    timestamps = ts_head.decode(start_logits, end_logits, frame_duration_ms=40)
    print(f"  Sample: {timestamps[0][0]}")

    # Confidence head
    conf_head = ConfidenceHead(hidden_dim).to(device)
    confidence = conf_head(hidden)
    print(
        f"✓ ConfidenceHead: {confidence.shape}, range=[{confidence.min():.2f}, {confidence.max():.2f}]"
    )

    return True


def test_full_model():
    """Test full model (requires ~2GB VRAM for 0.5B)."""
    print("\n" + "=" * 50)
    print("TEST 4: Full Model (0.5B)")
    print("=" * 50)

    from src.config import get_config
    from src.model import VoxLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check memory
    if device == "cuda":
        free_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {free_mem:.1f} GB")
        if free_mem < 4:
            print("Low GPU memory, model may not fit")

    print("\nLoading VoxLM-0.5B...")
    print("(This downloads ~1GB of model weights on first run)")

    config = get_config("voxlm-0.5b")
    print(f"  Audio Encoder: {config.audio_encoder}")
    print(f"  LLM: {config.llm_model}")

    try:
        model = VoxLM(config)
        model = model.to(device)
        print("Model loaded")

        # Print parameters
        model.print_trainable_parameters()

        # Test forward pass
        print("\nTesting forward pass...")
        test_audio = torch.randn(16000 * 3).to(device)  # 3 seconds

        with torch.no_grad():
            outputs = model(
                audio=test_audio.unsqueeze(0),
                instruction="Test transcription",
            )

        print(f"✓ Forward pass successful")
        print(f"  Start logits: {outputs.start_logits.shape}")
        print(f"  End logits: {outputs.end_logits.shape}")
        print(f"  Confidence: {outputs.confidence.shape}")

        return True

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_transcribe():
    """Test transcription function."""
    print("\n" + "=" * 50)
    print("TEST 5: Transcription")
    print("=" * 50)

    from src.config import get_config
    from src.model import VoxLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = get_config("voxlm-0.5b")
    model = VoxLM(config).to(device)

    # Create test audio
    test_audio = torch.randn(16000 * 2).to(device)  # 2 seconds

    print("Running transcription (untrained model - expect random output)...")

    try:
        result = model.transcribe(
            audio=test_audio,
            instruction="Clear English speech.",
            max_length=20,
        )

        print(f"✓ Transcription successful")
        print(f"  Text: {result['text'][:100]}...")
        print(f"  Words: {len(result.get('words', []))} words")
        print(f"  Duration: {result['audio_duration']:.2f}s")

        if result.get("words"):
            print(f"  First word: {result['words'][0]}")

        return True

    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n" + "#" * 50)
    print("# VoxLM Local Test Suite")
    print("#" * 50)

    # Check environment
    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    results = {}

    # Run tests
    results["imports"] = test_imports()

    if results["imports"]:
        results["audio_encoder"] = test_audio_encoder()
        results["heads"] = test_heads()

        # Only run full model test if user confirms (uses GPU memory)
        print("\n" + "-" * 50)
        run_full = (
            input("Run full model test? (requires ~4GB VRAM) [y/N]: ").lower().strip()
        )

        if run_full == "y":
            results["full_model"] = test_full_model()

            if results.get("full_model"):
                run_transcribe = (
                    input("Run transcription test? [y/N]: ").lower().strip()
                )
                if run_transcribe == "y":
                    results["transcribe"] = test_transcribe()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print(
        f"\nOverall: {'✓ All tests passed!' if all_passed else '✗ Some tests failed'}"
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
