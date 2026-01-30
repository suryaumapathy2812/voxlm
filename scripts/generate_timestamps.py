#!/usr/bin/env python3
"""
Generate word-level timestamps for LibriSpeech using WhisperX.

This script:
1. Loads LibriSpeech audio files
2. Runs WhisperX to get word-level timestamps
3. Validates against ground truth transcripts
4. Saves timestamps as JSON manifest

Usage:
    # Using YAML config (recommended)
    python scripts/generate_timestamps.py --config configs/voxlm-2b.yaml --split train-clean-100
    python scripts/generate_timestamps.py --config configs/voxlm-2b.yaml --split dev-clean

    # Using CLI arguments (legacy)
    python scripts/generate_timestamps.py --data ./data --split train-clean-100 --output ./data/timestamps
    python scripts/generate_timestamps.py --data ./data --split dev-clean --output ./data/timestamps
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

# Add src to path for config loading
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for whisperx
try:
    import whisperx

    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False
    print("WhisperX not installed. Install with: pip install whisperx")


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase, remove punctuation, normalize whitespace
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)  # Keep apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_error_rate(ref: str, hyp: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis."""
    ref_words = ref.split()
    hyp_words = hyp.split()

    # Simple Levenshtein distance at word level
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / max(m, 1)


def load_librispeech_samples(data_dir: Path, split: str) -> list:
    """Load LibriSpeech samples with ground truth transcripts."""
    samples = []

    # Try different paths
    split_dir = data_dir / split
    if not split_dir.exists():
        split_dir = data_dir / "LibriSpeech" / split

    if not split_dir.exists():
        print(f"ERROR: Split directory not found: {split_dir}")
        return []

    print(f"Loading samples from: {split_dir}")

    # Find all trans.txt files
    trans_files = list(split_dir.rglob("*.trans.txt"))
    print(f"Found {len(trans_files)} transcript files")

    for trans_file in trans_files:
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, text = parts
                    audio_path = trans_file.parent / f"{audio_id}.flac"
                    if audio_path.exists():
                        samples.append(
                            {
                                "id": audio_id,
                                "audio": str(audio_path.resolve()),
                                "text": text,  # Original case
                                "text_normalized": normalize_text(text),
                            }
                        )

    print(f"Loaded {len(samples)} samples")
    return samples


def generate_timestamps_whisperx(
    samples: list,
    output_dir: Path,
    whisper_model: str = "small",
    device: str = "cuda",
    batch_size: int = 16,
    max_wer: float = 0.1,  # Only keep samples with WER < 10%
) -> dict:
    """
    Generate timestamps using WhisperX.

    Args:
        samples: List of samples with audio paths and ground truth
        output_dir: Where to save results
        whisper_model: Whisper model size
        device: cuda or cpu
        batch_size: Batch size for WhisperX
        max_wer: Maximum WER to accept (filters bad alignments)

    Returns:
        Statistics dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load WhisperX model
    print(f"\nLoading WhisperX model: {whisper_model}")
    compute_type = "float16" if device == "cuda" else "float32"
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type)

    # Load alignment model
    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    # Process samples
    results = []
    stats = {
        "total": len(samples),
        "success": 0,
        "failed_wer": 0,
        "failed_error": 0,
    }

    print(f"\nProcessing {len(samples)} samples...")

    for sample in tqdm(samples):
        try:
            # Transcribe with WhisperX
            audio = whisperx.load_audio(sample["audio"])
            result = model.transcribe(audio, batch_size=batch_size)

            # Align to get word timestamps
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )

            # Extract words with timestamps
            words = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        words.append(
                            {
                                "word": word_info["word"],
                                "start": word_info.get("start", 0.0),
                                "end": word_info.get("end", 0.0),
                            }
                        )

            # Get WhisperX transcript
            whisperx_text = " ".join([w["word"] for w in words])
            whisperx_normalized = normalize_text(whisperx_text)

            # Calculate WER
            wer = word_error_rate(sample["text_normalized"], whisperx_normalized)

            if wer <= max_wer:
                # Good alignment - save it
                results.append(
                    {
                        "id": sample["id"],
                        "audio": sample["audio"],
                        "text": sample["text"].lower(),
                        "text_whisperx": whisperx_text.lower(),
                        "wer": round(wer, 4),
                        "timestamps": words,
                    }
                )
                stats["success"] += 1
            else:
                stats["failed_wer"] += 1

        except Exception as e:
            stats["failed_error"] += 1
            if stats["failed_error"] <= 5:  # Only print first 5 errors
                print(f"\nError processing {sample['id']}: {e}")

    # Save results
    manifest_path = output_dir / f"timestamps_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {stats['total']}")
    print(
        f"Successful: {stats['success']} ({100 * stats['success'] / stats['total']:.1f}%)"
    )
    print(f"Failed (WER > {max_wer}): {stats['failed_wer']}")
    print(f"Failed (errors): {stats['failed_error']}")
    print(f"\nManifest saved to: {manifest_path}")

    return stats


def generate_timestamps_whisper_native(
    samples: list,
    output_dir: Path,
    whisper_model: str = "small",
    device: str = "cuda",
    max_wer: float = 0.1,
) -> dict:
    """
    Fallback: Generate timestamps using native Whisper word_timestamps.

    Less accurate than WhisperX but no extra dependencies.
    """
    import whisper

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading Whisper model: {whisper_model}")
    model = whisper.load_model(whisper_model, device=device)

    results = []
    stats = {
        "total": len(samples),
        "success": 0,
        "failed_wer": 0,
        "failed_error": 0,
    }

    print(f"\nProcessing {len(samples)} samples...")

    for sample in tqdm(samples):
        try:
            # Transcribe with word timestamps
            result = model.transcribe(
                sample["audio"],
                word_timestamps=True,
                language="en",
            )

            # Extract words
            words = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        words.append(
                            {
                                "word": word_info["word"].strip(),
                                "start": word_info.get("start", 0.0),
                                "end": word_info.get("end", 0.0),
                            }
                        )

            # Get transcript
            whisper_text = " ".join([w["word"] for w in words])
            whisper_normalized = normalize_text(whisper_text)

            # Calculate WER
            wer = word_error_rate(sample["text_normalized"], whisper_normalized)

            if wer <= max_wer:
                results.append(
                    {
                        "id": sample["id"],
                        "audio": sample["audio"],
                        "text": sample["text"].lower(),
                        "text_whisper": whisper_text.lower(),
                        "wer": round(wer, 4),
                        "timestamps": words,
                    }
                )
                stats["success"] += 1
            else:
                stats["failed_wer"] += 1

        except Exception as e:
            stats["failed_error"] += 1
            if stats["failed_error"] <= 5:
                print(f"\nError processing {sample['id']}: {e}")

    # Save results
    manifest_path = output_dir / f"timestamps_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {stats['total']}")
    print(
        f"Successful: {stats['success']} ({100 * stats['success'] / stats['total']:.1f}%)"
    )
    print(f"Failed (WER > {max_wer}): {stats['failed_wer']}")
    print(f"Failed (errors): {stats['failed_error']}")
    print(f"\nManifest saved to: {manifest_path}")

    return stats


def generate_timestamps_whisper_batched(
    samples: list,
    output_dir: Path,
    whisper_model: str = "turbo",
    device: str = "cuda",
    max_wer: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict:
    """
    FAST: Generate timestamps using Whisper with parallel audio loading.

    Uses multiprocessing to load/preprocess audio while GPU processes previous batch.
    """
    import whisper
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading Whisper model: {whisper_model}")
    model = whisper.load_model(whisper_model, device=device)

    results = []
    stats = {
        "total": len(samples),
        "success": 0,
        "failed_wer": 0,
        "failed_error": 0,
    }

    def load_audio_for_whisper(sample):
        """Load and preprocess audio for Whisper."""
        try:
            audio = whisper.load_audio(sample["audio"])
            return sample, audio, None
        except Exception as e:
            return sample, None, str(e)

    print(f"\nProcessing {len(samples)} samples with {num_workers} workers...")

    # Process in batches with prefetching
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all audio loading tasks
        futures = {executor.submit(load_audio_for_whisper, s): s for s in samples}

        pbar = tqdm(total=len(samples))
        for future in as_completed(futures):
            sample, audio, error = future.result()

            if error:
                stats["failed_error"] += 1
                if stats["failed_error"] <= 5:
                    print(f"\nError loading {sample['id']}: {error}")
                pbar.update(1)
                continue

            try:
                # Transcribe with word timestamps
                result = model.transcribe(
                    audio,
                    word_timestamps=True,
                    language="en",
                )

                # Extract words
                words = []
                for segment in result["segments"]:
                    if "words" in segment:
                        for word_info in segment["words"]:
                            words.append(
                                {
                                    "word": word_info["word"].strip(),
                                    "start": word_info.get("start", 0.0),
                                    "end": word_info.get("end", 0.0),
                                }
                            )

                # Get transcript
                whisper_text = " ".join([w["word"] for w in words])
                whisper_normalized = normalize_text(whisper_text)

                # Calculate WER
                wer = word_error_rate(sample["text_normalized"], whisper_normalized)

                if wer <= max_wer:
                    results.append(
                        {
                            "id": sample["id"],
                            "audio": sample["audio"],
                            "text": sample["text"].lower(),
                            "text_whisper": whisper_text.lower(),
                            "wer": round(wer, 4),
                            "timestamps": words,
                        }
                    )
                    stats["success"] += 1
                else:
                    stats["failed_wer"] += 1

            except Exception as e:
                stats["failed_error"] += 1
                if stats["failed_error"] <= 5:
                    print(f"\nError processing {sample['id']}: {e}")

            pbar.update(1)

        pbar.close()

    # Save results
    manifest_path = output_dir / f"timestamps_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {stats['total']}")
    print(
        f"Successful: {stats['success']} ({100 * stats['success'] / stats['total']:.1f}%)"
    )
    print(f"Failed (WER > {max_wer}): {stats['failed_wer']}")
    print(f"Failed (errors): {stats['failed_error']}")
    print(f"\nManifest saved to: {manifest_path}")

    return stats


def _worker_fn(
    worker_id: int,
    sample_chunk: list,
    result_queue,
    device: str,
    whisper_model: str,
    max_wer: float,
):
    """Worker process that runs Whisper on a chunk of samples."""
    import whisper

    # Each worker loads its own model
    print(f"Worker {worker_id}: Loading Whisper model...")
    model = whisper.load_model(whisper_model, device=device)

    results = []
    stats = {"success": 0, "failed_wer": 0, "failed_error": 0}

    for i, sample in enumerate(sample_chunk):
        try:
            result = model.transcribe(
                sample["audio"],
                word_timestamps=True,
                language="en",
            )

            words = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        words.append(
                            {
                                "word": word_info["word"].strip(),
                                "start": word_info.get("start", 0.0),
                                "end": word_info.get("end", 0.0),
                            }
                        )

            whisper_text = " ".join([w["word"] for w in words])
            whisper_normalized = normalize_text(whisper_text)
            wer = word_error_rate(sample["text_normalized"], whisper_normalized)

            if wer <= max_wer:
                results.append(
                    {
                        "id": sample["id"],
                        "audio": sample["audio"],
                        "text": sample["text"].lower(),
                        "text_whisper": whisper_text.lower(),
                        "wer": round(wer, 4),
                        "timestamps": words,
                    }
                )
                stats["success"] += 1
            else:
                stats["failed_wer"] += 1

        except Exception as e:
            stats["failed_error"] += 1

        # Progress update every 100 samples
        if (i + 1) % 100 == 0:
            print(f"Worker {worker_id}: {i + 1}/{len(sample_chunk)} samples processed")

    result_queue.put((worker_id, results, stats))
    print(f"Worker {worker_id}: Done! Processed {len(sample_chunk)} samples")


def generate_timestamps_multiprocess(
    samples: list,
    output_dir: Path,
    whisper_model: str = "turbo",
    device: str = "cuda",
    max_wer: float = 0.1,
    num_processes: int = 4,
) -> dict:
    """
    TRUE PARALLEL: Run multiple Whisper processes on GPU.

    Splits samples across N processes, each with its own Whisper model instance.
    H100 can handle 4-6 turbo models simultaneously.
    """
    import multiprocessing as mp

    # CRITICAL: Use 'spawn' for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split samples across workers
    chunk_size = len(samples) // num_processes
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else len(samples)
        chunks.append(samples[start:end])

    print(f"\nStarting {num_processes} parallel workers...")
    print(f"Each worker processes ~{chunk_size} samples")

    # Start workers
    result_queue = mp.Queue()
    processes = []

    for i, chunk in enumerate(chunks):
        p = mp.Process(
            target=_worker_fn,
            args=(i, chunk, result_queue, device, whisper_model, max_wer),
        )
        p.start()
        processes.append(p)

    # Collect results with progress
    all_results = []
    total_stats = {"success": 0, "failed_wer": 0, "failed_error": 0}

    for _ in range(num_processes):
        worker_id, results, stats = result_queue.get()
        all_results.extend(results)
        for k in total_stats:
            total_stats[k] += stats[k]
        print(f"Collected results from worker {worker_id}")

    # Wait for all processes
    for p in processes:
        p.join()

    # Save results
    manifest_path = output_dir / "timestamps_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(samples)}")
    print(
        f"Successful: {total_stats['success']} ({100 * total_stats['success'] / len(samples):.1f}%)"
    )
    print(f"Failed (WER > {max_wer}): {total_stats['failed_wer']}")
    print(f"Failed (errors): {total_stats['failed_error']}")
    print(f"\nManifest saved to: {manifest_path}")

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate word timestamps for LibriSpeech"
    )

    # YAML config support
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/voxlm-2b.yaml). CLI args override config values.",
    )

    # Data arguments (can be overridden by CLI)
    parser.add_argument("--data", type=str, default=None, help="Path to data directory")
    parser.add_argument("--split", type=str, default=None, help="LibriSpeech split")
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory for timestamps"
    )
    parser.add_argument("--model", type=str, default=None, help="Whisper model size")
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda, cpu, auto"
    )
    parser.add_argument(
        "--max-wer",
        type=float,
        default=None,
        help="Max WER to accept (0.1 = 10 percent)",
    )
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Use native Whisper instead of WhisperX",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for WhisperX"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast parallel audio loading (recommended)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers for parallel loading",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel Whisper processes (0=disabled, 4-6 recommended for H100)",
    )

    args = parser.parse_args()

    # ==========================================================================
    # Load configuration from YAML or use CLI defaults
    # ==========================================================================
    if args.config:
        print(f"Loading configuration from: {args.config}")
        # Import here to avoid loading torch at module level
        from src.config import load_full_config

        full_config = load_full_config(args.config)

        # Extract values from YAML config (CLI args override these)
        data_dir = args.data or full_config.data.data_dir
        split = args.split or full_config.data.train_split  # Default to train split
        output_dir = args.output or full_config.data.timestamps_dir
        whisper_model = args.model or full_config.data.whisper_model
        device = args.device or full_config.device
        max_wer = args.max_wer if args.max_wer is not None else full_config.data.max_wer
        batch_size = args.batch_size if args.batch_size is not None else 16
        num_workers = args.num_workers if args.num_workers is not None else 8
        parallel = args.parallel if args.parallel is not None else 0
    else:
        # Use CLI arguments with defaults
        data_dir = args.data or "./data"
        split = args.split or "train-clean-100"
        output_dir = args.output or "./data/timestamps"
        whisper_model = args.model or "small"
        device = args.device or "auto"
        max_wer = args.max_wer if args.max_wer is not None else 0.1
        batch_size = args.batch_size if args.batch_size is not None else 16
        num_workers = args.num_workers if args.num_workers is not None else 8
        parallel = args.parallel if args.parallel is not None else 0

    # Auto device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load samples
    data_path = Path(data_dir)
    samples = load_librispeech_samples(data_path, split)

    if not samples:
        print("No samples found!")
        return

    # Create output directory with split name
    output_path = Path(output_dir) / split

    # Generate timestamps
    if parallel > 0:
        # TRUE PARALLEL: Multiple Whisper processes
        print(f"Using TRUE PARALLEL mode with {parallel} processes")
        generate_timestamps_multiprocess(
            samples=samples,
            output_dir=output_path,
            whisper_model=whisper_model,
            device=device,
            max_wer=max_wer,
            num_processes=parallel,
        )
    elif args.fast:
        # Fast mode with parallel audio loading
        print("Using FAST mode with parallel audio loading")
        generate_timestamps_whisper_batched(
            samples=samples,
            output_dir=output_path,
            whisper_model=whisper_model,
            device=device,
            max_wer=max_wer,
            num_workers=num_workers,
        )
    elif args.use_native or not HAS_WHISPERX:
        if not HAS_WHISPERX:
            print("WhisperX not available, using native Whisper")
        generate_timestamps_whisper_native(
            samples=samples,
            output_dir=output_path,
            whisper_model=whisper_model,
            device=device,
            max_wer=max_wer,
        )
    else:
        generate_timestamps_whisperx(
            samples=samples,
            output_dir=output_path,
            whisper_model=whisper_model,
            device=device,
            batch_size=batch_size,
            max_wer=max_wer,
        )


if __name__ == "__main__":
    main()
