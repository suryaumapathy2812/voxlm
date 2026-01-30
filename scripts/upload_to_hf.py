#!/usr/bin/env python3
"""
Upload VoxLM model to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py \
        --checkpoint models/voxlm-2b/model.pt \
        --repo-id your-username/voxlm-2b \
        --private  # optional, makes repo private

Requirements:
    pip install huggingface_hub
    huggingface-cli login  # authenticate first
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)

import torch


def prepare_model_files(checkpoint_path: str, output_dir: str) -> Path:
    """
    Prepare model files for upload.

    Creates a directory with:
    - model.pt (the checkpoint)
    - config.json (model configuration)
    - README.md (model card)
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint to extract config
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Copy checkpoint
    output_checkpoint = output_dir / "model.pt"
    shutil.copy(checkpoint_path, output_checkpoint)
    print(f"Copied checkpoint to: {output_checkpoint}")

    # Save config as JSON
    config = checkpoint.get("config")
    if config:
        import json

        config_dict = {
            "model_name": getattr(config, "model_name", "voxlm-2b"),
            "whisper_model": getattr(config, "whisper_model", "openai/whisper-small"),
            "llm_model": getattr(config, "llm_model", "Qwen/Qwen2-1.5B"),
            "whisper_dim": getattr(config, "whisper_dim", 768),
            "llm_dim": getattr(config, "llm_dim", 1536),
            "downsample_factor": getattr(config, "downsample_factor", 4),
            "use_lora": getattr(config, "use_lora", True),
            "lora_r": getattr(config, "lora_r", 16),
            "architecture_version": getattr(config, "architecture_version", "v2"),
        }
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Saved config to: {config_path}")

    # Copy README if exists
    readme_src = checkpoint_path.parent / "README.md"
    if readme_src.exists():
        shutil.copy(readme_src, output_dir / "README.md")
        print(f"Copied README from: {readme_src}")
    else:
        # Check models directory
        models_readme = Path("models/voxlm-2b/README.md")
        if models_readme.exists():
            shutil.copy(models_readme, output_dir / "README.md")
            print(f"Copied README from: {models_readme}")

    # Add training info
    training_info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "global_step": checkpoint.get("global_step", "unknown"),
        "val_loss": checkpoint.get("val_loss", "unknown"),
    }
    info_path = output_dir / "training_info.json"
    import json

    with open(info_path, "w") as f:
        json.dump(training_info, f, indent=2)
    print(f"Saved training info to: {info_path}")

    return output_dir


def upload_to_hub(
    model_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload VoxLM model",
):
    """Upload model directory to Hugging Face Hub."""
    api = HfApi()

    # Create repo if it doesn't exist
    print(f"\nCreating/accessing repo: {repo_id}")
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload files
    print(f"\nUploading files from: {model_dir}")
    upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        commit_message=commit_message,
    )

    print(f"\n{'=' * 50}")
    print(f"Upload complete!")
    print(f"Model URL: https://huggingface.co/{repo_id}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Upload VoxLM model to Hugging Face")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (model.pt)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., username/voxlm-2b)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".hf_upload_temp",
        help="Temporary directory for prepared files",
    )
    args = parser.parse_args()

    # Prepare files
    print("Preparing model files for upload...")
    model_dir = prepare_model_files(args.checkpoint, args.output_dir)

    # Upload
    upload_to_hub(
        model_dir=str(model_dir),
        repo_id=args.repo_id,
        private=args.private,
    )

    # Cleanup temp dir
    print(f"\nCleaning up temp directory: {args.output_dir}")
    shutil.rmtree(args.output_dir, ignore_errors=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
