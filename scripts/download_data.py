#!/usr/bin/env python3
"""
Download LibriSpeech dataset for Qwen-STT training.

Usage:
    python scripts/download_data.py --subset dev-clean      # 337M, for testing
    python scripts/download_data.py --subset train-clean-100  # 6.3G, for training
"""

import argparse
import os
from pathlib import Path

# Available subsets and their sizes
SUBSETS = {
    "dev-clean": {"size": "337M", "hours": 5, "desc": "Development set, clean speech"},
    "dev-other": {"size": "314M", "hours": 5, "desc": "Development set, challenging"},
    "test-clean": {"size": "346M", "hours": 5, "desc": "Test set, clean speech"},
    "test-other": {"size": "328M", "hours": 5, "desc": "Test set, challenging"},
    "train-clean-100": {"size": "6.3G", "hours": 100, "desc": "Training, 100h clean"},
    "train-clean-360": {"size": "23G", "hours": 360, "desc": "Training, 360h clean"},
    "train-other-500": {"size": "30G", "hours": 500, "desc": "Training, 500h other"},
}


def download_with_torchaudio(subset: str, data_dir: str):
    """Download using torchaudio (recommended)."""
    import torchaudio
    from torchaudio.datasets import LIBRISPEECH

    print(f"Downloading {subset} using torchaudio...")
    print(f"Destination: {data_dir}")

    dataset = LIBRISPEECH(
        root=data_dir,
        url=subset,
        download=True,
    )

    print(f"✓ Downloaded {len(dataset)} samples")
    return dataset


def download_with_wget(subset: str, data_dir: str):
    """Download using wget (fallback)."""
    import subprocess

    base_url = "https://www.openslr.org/resources/12"
    filename = f"{subset}.tar.gz"
    url = f"{base_url}/{filename}"

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    filepath = data_dir / filename

    print(f"Downloading {url}...")
    subprocess.run(["wget", "-c", url, "-O", str(filepath)], check=True)

    print(f"Extracting {filepath}...")
    subprocess.run(["tar", "-xzf", str(filepath), "-C", str(data_dir)], check=True)

    # Optionally remove tar file
    # filepath.unlink()

    print(f"✓ Extracted to {data_dir}/LibriSpeech/{subset}")


def create_manifest(subset: str, data_dir: str):
    """Create a JSON manifest for the dataset."""
    import json

    data_dir = Path(data_dir)
    subset_dir = data_dir / "LibriSpeech" / subset

    if not subset_dir.exists():
        print(f"Directory not found: {subset_dir}")
        return

    samples = []

    # Find all transcription files
    for trans_file in subset_dir.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, text = parts
                    audio_path = trans_file.parent / f"{audio_id}.flac"

                    if audio_path.exists():
                        samples.append({
                            "audio": str(audio_path),
                            "text": text.lower(),
                            "id": audio_id,
                        })

    # Save manifest
    manifest_path = data_dir / f"{subset}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"✓ Created manifest: {manifest_path}")
    print(f"  Total samples: {len(samples)}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Download LibriSpeech data")
    parser.add_argument(
        "--subset",
        type=str,
        default="dev-clean",
        choices=list(SUBSETS.keys()),
        help="Which subset to download",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Where to save the data",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="torchaudio",
        choices=["torchaudio", "wget"],
        help="Download method",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available subsets",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable LibriSpeech subsets:")
        print("-" * 60)
        for name, info in SUBSETS.items():
            print(f"  {name:20s} {info['size']:>6s}  {info['hours']:>4d}h  {info['desc']}")
        print("\nRecommended for testing: dev-clean (337M)")
        print("Recommended for training: train-clean-100 (6.3G)")
        return

    info = SUBSETS[args.subset]
    print(f"\nDownloading: {args.subset}")
    print(f"  Size: {info['size']}")
    print(f"  Hours: {info['hours']}")
    print(f"  Description: {info['desc']}")
    print()

    # Download
    if args.method == "torchaudio":
        try:
            download_with_torchaudio(args.subset, args.data_dir)
        except Exception as e:
            print(f"torchaudio failed: {e}")
            print("Falling back to wget...")
            download_with_wget(args.subset, args.data_dir)
    else:
        download_with_wget(args.subset, args.data_dir)

    # Create manifest
    create_manifest(args.subset, args.data_dir)

    print("\n✓ Done!")
    print(f"\nTo train with this data:")
    print(f"  python scripts/train.py --model qwen-stt-0.5b --data {args.data_dir}")


if __name__ == "__main__":
    main()
