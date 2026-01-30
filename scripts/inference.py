#!/usr/bin/env python3
"""
Quick inference script to test trained VoxLM model.

Usage:
    # Using YAML config (recommended)
    python scripts/inference.py --config configs/voxlm-2b.yaml --audio path/to/audio.flac
    python scripts/inference.py --config configs/voxlm-2b.yaml --checkpoint path/to/model.pt --audio path/to/audio.flac

    # Using CLI arguments (legacy)
    python scripts/inference.py --checkpoint checkpoints/voxlm-2b/best.pt --audio data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac
    python scripts/inference.py --checkpoint checkpoints/voxlm-2b/best.pt --audio <path_to_audio>

    # Legacy checkpoint paths still work
    python scripts/inference.py --checkpoint checkpoints/qwen-stt-2b/best.pt --audio <path_to_audio>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import json
import soundfile as sf
import numpy as np

from src.config import get_config, load_full_config, load_config_from_yaml
from src.model import VoxLM


def load_audio(path, target_sr=16000):
    """Load and preprocess audio file using soundfile."""
    # Load with soundfile
    audio, sr = sf.read(path)

    # Convert to torch tensor
    waveform = torch.from_numpy(audio.astype(np.float32))

    # Handle stereo -> mono
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=-1)

    # Resample if needed
    if sr != target_sr:
        import torchaudio

        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    return waveform, target_sr


def main():
    parser = argparse.ArgumentParser(description="Test VoxLM inference")

    # YAML config support
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/voxlm-2b.yaml). CLI args override config values.",
    )

    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument(
        "--instruction", type=str, default=None, help="Instruction for transcription"
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--list-samples", action="store_true", help="List sample audio files"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda, cpu, or auto"
    )

    args = parser.parse_args()

    # ==========================================================================
    # Load configuration from YAML or use CLI defaults
    # ==========================================================================
    if args.config:
        print(f"Loading configuration from: {args.config}")
        full_config = load_full_config(args.config)

        # Build default checkpoint path from config
        default_checkpoint = (
            f"{full_config.output.final_model_dir}/{full_config.model.name}/model.pt"
        )
        # Fallback to checkpoint_dir if final model doesn't exist
        if not Path(default_checkpoint).exists():
            default_checkpoint = f"{full_config.output.checkpoint_dir}/{full_config.model.name}/best_phase2.pt"
        if not Path(default_checkpoint).exists():
            default_checkpoint = (
                f"{full_config.output.checkpoint_dir}/{full_config.model.name}/best.pt"
            )

        # Extract values from YAML config (CLI args override these)
        checkpoint_path = args.checkpoint or default_checkpoint
        device = args.device or full_config.device
        instruction = args.instruction or "Transcribe this audio clearly."
        max_length = args.max_length if args.max_length is not None else 200

        # Store model config for later use
        yaml_model_config = full_config.model
    else:
        # Use CLI arguments with defaults
        checkpoint_path = args.checkpoint or "checkpoints/voxlm-2b/best.pt"
        device = args.device or "auto"
        instruction = args.instruction or "Transcribe this audio clearly."
        max_length = args.max_length if args.max_length is not None else 200
        yaml_model_config = None

    # List sample files if requested
    if args.list_samples:
        sample_dir = Path("data/LibriSpeech/dev-clean")
        if sample_dir.exists():
            print("\nSample audio files:")
            for i, flac in enumerate(sample_dir.rglob("*.flac")):
                if i >= 10:
                    print("  ...")
                    break
                print(f"  {flac}")
        return

    if not args.audio:
        # Use a default sample
        default_samples = list(Path("data/LibriSpeech/dev-clean").rglob("*.flac"))[:1]
        if default_samples:
            audio_path_str = str(default_samples[0])
            print(f"Using default sample: {audio_path_str}")
        else:
            print("No audio file specified. Use --audio <path> or --list-samples")
            return
    else:
        audio_path_str = args.audio

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model config (priority: checkpoint > YAML config > default)
    config = checkpoint.get("config")
    if config is None and yaml_model_config is not None:
        config = yaml_model_config
    if config is None:
        config = get_config("voxlm-2b")

    model = VoxLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Load audio
    print(f"\nLoading audio: {audio_path_str}")
    audio, sr = load_audio(audio_path_str)
    duration = len(audio) / sr
    print(f"Duration: {duration:.2f}s")

    # Get ground truth if available (LibriSpeech format)
    audio_path = Path(audio_path_str)
    trans_file = audio_path.parent / f"{audio_path.parent.name}.trans.txt"
    ground_truth = None
    if trans_file.exists():
        audio_id = audio_path.stem
        with open(trans_file) as f:
            for line in f:
                if line.startswith(audio_id):
                    ground_truth = line.strip().split(" ", 1)[1].lower()
                    break

    # Transcribe
    print(f"\nInstruction: {instruction}")
    print("Transcribing...")

    with torch.no_grad():
        result = model.transcribe(
            audio=audio.to(device),
            instruction=instruction,
            max_length=max_length,
        )

    # Post-process: Clamp timestamps to audio duration
    if result.get("words"):
        words = result["words"]

        # Check if timestamps are broken (beyond audio duration or all zeros)
        max_end = max(w.get("end", 0) or 0 for w in words)
        all_zeros = all(
            (w.get("start", 0) or 0) == 0 and (w.get("end", 0) or 0) == 0 for w in words
        )

        if max_end > duration * 1.5 or all_zeros:
            # Timestamps are broken - distribute evenly across audio duration
            print(
                f"\n[INFO] Timestamps appear broken (max_end={max_end:.2f}s, duration={duration:.2f}s)"
            )
            print("[INFO] Applying fallback: distributing timestamps evenly")

            num_words = len(words)
            word_duration = duration / num_words

            for i, word in enumerate(words):
                word["start"] = round(i * word_duration, 3)
                word["end"] = round((i + 1) * word_duration, 3)
        else:
            # Just clamp to audio duration
            for word in words:
                if word.get("start") is not None:
                    word["start"] = min(max(word["start"], 0), duration)
                if word.get("end") is not None:
                    word["end"] = min(max(word["end"], 0), duration)
                # Ensure end >= start
                if word.get("start") is not None and word.get("end") is not None:
                    if word["end"] < word["start"]:
                        word["end"] = word["start"]

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if ground_truth:
        print(f"\nGround Truth: {ground_truth}")

    print(f"\nPredicted: {result['text']}")
    print(f"\nAudio Duration: {result['audio_duration']:.2f}s")

    if result.get("words"):
        print(f"\nWord Timestamps ({len(result['words'])} words):")
        for i, word in enumerate(result["words"][:20]):  # Show first 20
            conf = word.get("confidence", 0.0)
            start = word.get("start", 0.0) or 0.0
            end = word.get("end", 0.0) or 0.0
            print(f"  {word['word']:15s} {start:.2f}s - {end:.2f}s (conf: {conf:.2f})")
        if len(result["words"]) > 20:
            print(f"  ... and {len(result['words']) - 20} more words")

    # Save full result
    output_path = "inference_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to: {output_path}")


if __name__ == "__main__":
    main()
