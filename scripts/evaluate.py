#!/usr/bin/env python3
"""
Evaluation script for VoxLM models.

Computes Word Error Rate (WER) and other metrics on a test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/model/best.pt --split dev-clean
    python scripts/evaluate.py --checkpoint output/models/voxlm-2b/model.pt --split test-clean
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import torch
import soundfile as sf
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import VoxLM
from src.config import get_config


def compute_wer(reference: str, hypothesis: str) -> Dict:
    """
    Compute Word Error Rate between reference and hypothesis.

    WER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = words in reference
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    edit_distance = d[len(ref_words)][len(hyp_words)]
    wer = edit_distance / max(len(ref_words), 1)

    return {
        "wer": wer,
        "edit_distance": edit_distance,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
    }


def load_librispeech_samples(
    data_dir: str, split: str, limit: int = None
) -> List[Dict]:
    """Load LibriSpeech samples for evaluation."""
    samples = []
    split_dir = Path(data_dir) / "LibriSpeech" / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Find all transcript files
    for trans_file in split_dir.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, text = parts
                    audio_path = trans_file.parent / f"{audio_id}.flac"
                    if audio_path.exists():
                        samples.append(
                            {
                                "audio_path": str(audio_path),
                                "reference": text.lower(),
                                "id": audio_id,
                            }
                        )

                        if limit and len(samples) >= limit:
                            return samples

    return samples


def evaluate_model(
    model: VoxLM,
    samples: List[Dict],
    device: str = "cuda",
) -> Dict:
    """
    Evaluate model on samples.

    Returns:
        Dict with WER and other metrics
    """
    model.eval()

    all_wer = []
    all_results = []
    eos_generated_count = 0

    for sample in tqdm(samples, desc="Evaluating"):
        # Load audio
        audio, sr = sf.read(sample["audio_path"])
        if sr != 16000:
            # Simple resampling (for proper resampling, use torchaudio)
            import numpy as np

            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * 16000 / sr)),
                np.arange(len(audio)),
                audio,
            )

        audio_tensor = torch.from_numpy(audio).float().to(device)

        # Transcribe
        with torch.no_grad():
            result = model.transcribe(audio_tensor)

        hypothesis = result["text"]
        reference = sample["reference"]

        # Compute WER
        wer_result = compute_wer(reference, hypothesis)
        all_wer.append(wer_result["wer"])

        # Check if EOS was generated (important for the fix!)
        # We check by seeing if the model stopped before max_length
        words_generated = len(hypothesis.split())
        audio_duration = len(audio) / 16000
        expected_max_words = int(audio_duration * 5)  # Conservative estimate
        eos_likely_generated = words_generated < expected_max_words
        if eos_likely_generated:
            eos_generated_count += 1

        all_results.append(
            {
                "id": sample["id"],
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer_result["wer"],
                "eos_likely": eos_likely_generated,
            }
        )

    # Aggregate metrics
    avg_wer = sum(all_wer) / len(all_wer) if all_wer else 0
    eos_rate = eos_generated_count / len(samples) if samples else 0

    return {
        "wer": avg_wer,
        "eos_generation_rate": eos_rate,
        "num_samples": len(samples),
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VoxLM model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev-clean",
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda, cpu, or auto",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for detailed results",
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config") or get_config("voxlm-2b")

    model = VoxLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Verify tokenizer setup
    print(f"\nTokenizer setup:")
    print(f"  eos_token_id: {model.tokenizer.eos_token_id}")
    print(f"  pad_token_id: {model.tokenizer.pad_token_id}")
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        print("  WARNING: pad_token == eos_token (model may not stop properly)")
    else:
        print("  OK: pad_token != eos_token")

    # Load samples
    print(f"\nLoading samples from: {args.data}/{args.split}")
    samples = load_librispeech_samples(args.data, args.split, args.limit)
    print(f"  Found {len(samples)} samples")

    # Evaluate
    print(f"\nEvaluating...")
    results = evaluate_model(model, samples, device)

    # Print results
    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 50}")
    print(f"  Split: {args.split}")
    print(f"  Samples: {results['num_samples']}")
    print(f"  WER: {results['wer'] * 100:.2f}%")
    print(f"  EOS Generation Rate: {results['eos_generation_rate'] * 100:.1f}%")
    print(f"{'=' * 50}")

    # Show some examples
    print(f"\nSample Results:")
    for r in results["results"][:5]:
        print(f"\n  ID: {r['id']}")
        print(f"  REF: {r['reference'][:80]}...")
        print(f"  HYP: {r['hypothesis'][:80]}...")
        print(f"  WER: {r['wer'] * 100:.1f}%")

    # Save detailed results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
