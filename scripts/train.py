#!/usr/bin/env python3
"""
Training script for VoxLM models.

Optimized for H100 GPUs with:
- Mixed precision (bfloat16 AMP)
- Gradient accumulation
- torch.compile() optimization
- TF32 tensor cores
- Optimized DataLoader

Usage:
    # Using YAML config (recommended)
    python scripts/train.py --config configs/voxlm-2b.yaml
    python scripts/train.py --config configs/voxlm-9b-global.yaml

    # Using CLI arguments (legacy)
    python scripts/train.py --model voxlm-2b --data ./data --epochs 10
    python scripts/train.py --model voxlm-9b-global --data ./data --batch-size 4

    # Mix: YAML config with CLI overrides
    python scripts/train.py --config configs/voxlm-2b.yaml --epochs 20 --lr 5e-5

    # With gradient accumulation for larger effective batch size
    python scripts/train.py --model voxlm-2b --batch-size 16 --grad-accum 4  # effective=64

    # Legacy model names still work for backward compatibility
    python scripts/train.py --model qwen-stt-2b --data ./data
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torchaudio
from tqdm import tqdm

# Optional wandb
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


def setup_torch_optimizations():
    """Enable PyTorch optimizations for H100/modern GPUs."""
    if torch.cuda.is_available():
        # Enable TF32 for matmul and cuDNN (significant speedup on Ampere/Hopper)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN autotuner (finds fastest algorithms)
        torch.backends.cudnn.benchmark = True

        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(
            f"TF32 enabled: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}"
        )
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    get_config,
    MODEL_CONFIGS,
    load_full_config,
    load_config_from_yaml,
)
from src.model import VoxLM


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset with word-level timestamps.

    Expects data in format:
    - audio files (.flac or .wav)
    - transcripts with word timestamps (from MFA or similar)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train-clean-100",
        max_audio_length: int = 30,
        sample_rate: int = 16000,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.max_samples = max_audio_length * sample_rate

        # Load manifest (you'll need to create this from LibriSpeech)
        self.samples = self._load_manifest()

    def _load_manifest(self):
        """Load dataset manifest."""
        print(f"DEBUG: Loading dataset for split='{self.split}'")
        print(f"DEBUG: data_dir={self.data_dir}")

        manifest_path = self.data_dir / f"{self.split}_manifest.json"

        if manifest_path.exists():
            print(f"DEBUG: Found manifest at {manifest_path}")
            with open(manifest_path) as f:
                return json.load(f)

        # If no manifest, try to build from directory structure
        samples = []

        # Try direct path first
        split_dir = self.data_dir / self.split
        print(f"DEBUG: Trying direct path: {split_dir}")
        print(f"DEBUG: split_dir.exists() = {split_dir.exists()}")

        # Also check LibriSpeech subdirectory (common extraction structure)
        if not split_dir.exists():
            split_dir = self.data_dir / "LibriSpeech" / self.split
            print(f"DEBUG: Trying LibriSpeech path: {split_dir}")
            print(f"DEBUG: split_dir.exists() = {split_dir.exists()}")

        # Final check - if still doesn't exist, use dummy data
        if not split_dir.exists():
            print(f"\n{'=' * 60}")
            print(f"ERROR: {self.split} not found in data directory.")
            print(f"{'=' * 60}")
            print(f"  Checked paths:")
            print(f"    1. {self.data_dir / self.split}")
            print(f"    2. {self.data_dir / 'LibriSpeech' / self.split}")
            print(f"\n  Available directories in {self.data_dir}:")
            for d in sorted(self.data_dir.iterdir()):
                print(f"    - {d.name}")

            librispeech_dir = self.data_dir / "LibriSpeech"
            if librispeech_dir.exists():
                print(f"\n  Available directories in data/LibriSpeech:")
                for d in sorted(librispeech_dir.iterdir()):
                    if d.is_dir() and not d.name.startswith("."):
                        print(f"    - {d.name}")

            # Create dummy samples for pipeline testing
            print(f"\nUsing dummy data for testing (10 samples).")
            print(f"{'=' * 60}\n")
            dummy_samples = []
            for i in range(10):
                dummy_samples.append(
                    {
                        "audio": None,  # Will generate random audio
                        "text": f"this is dummy sample number {i} for testing the pipeline",
                        "timestamps": [],
                        "is_dummy": True,
                    }
                )
            return dummy_samples

        print(f"DEBUG: Found split_dir = {split_dir}")
        print(f"DEBUG: Checking for *.trans.txt files in {split_dir}")

        # Build samples from trans.txt files
        trans_files = list(split_dir.rglob("*.trans.txt"))
        print(f"DEBUG: Found {len(trans_files)} trans.txt files")

        if not trans_files:
            print(f"\n{'=' * 60}")
            print(f"ERROR: No *.trans.txt files found in {split_dir}")
            print(f"{'=' * 60}")
            print(f"  Available files in {split_dir}:")
            files = list(split_dir.iterdir())
            if files:
                for f in sorted(files)[:10]:
                    print(f"    - {f.name}")
                if len(files) > 10:
                    print(f"    ... and {len(files) - 10} more")
            else:
                print(f"    (empty directory)")
            print(f"{'=' * 60}\n")
            return []

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
                                    "audio": str(
                                        audio_path.resolve()
                                    ),  # Use absolute path
                                    "text": text.lower(),
                                    "timestamps": [],  # Will need MFA alignment
                                }
                            )

        print(f"DEBUG: Built {len(samples)} samples")
        print(f"  Loaded {len(samples)} samples from {split_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Handle dummy data
        if sample.get("is_dummy", False):
            waveform = torch.randn(self.max_samples) * 0.1
            return {
                "audio": waveform,
                "text": sample["text"],
                "timestamps": [],
            }

        # Load audio using soundfile directly (more compatible than torchaudio)
        try:
            import soundfile as sf

            audio_data, sr = sf.read(sample["audio"])

            # Convert to torch tensor
            waveform = torch.from_numpy(audio_data).float()

            # Handle stereo -> mono
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=1)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

            # Truncate or pad
            if waveform.shape[0] > self.max_samples:
                waveform = waveform[: self.max_samples]
            elif waveform.shape[0] < self.max_samples:
                pad = torch.zeros(self.max_samples - waveform.shape[0])
                waveform = torch.cat([waveform, pad])

        except Exception as e:
            print(f"Error loading {sample['audio']}: {e}")
            waveform = torch.zeros(self.max_samples)

        return {
            "audio": waveform,
            "text": sample["text"],
            "timestamps": sample.get("timestamps", []),
        }


class CollateFn:
    """Picklable collate function for DataLoader multiprocessing."""

    def __init__(self, tokenizer, max_text_length=448, add_eos=True):
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.add_eos = add_eos

        # Ensure tokenizer has eos_token
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"

    def __call__(self, batch):
        """Collate batch with proper padding."""
        audios = torch.stack([b["audio"] for b in batch])

        # Tokenize texts - add EOS to teach model when to stop (fixes hallucination)
        texts = [b["text"] for b in batch]
        if self.add_eos:
            texts = [t + self.tokenizer.eos_token for t in texts]

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        return {
            "audio": audios,
            "labels": tokens.input_ids,
            "texts": texts,
        }


class Trainer:
    """Training loop for VoxLM with H100 optimizations."""

    def __init__(
        self,
        model: VoxLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str = "cuda",
        output_dir: str = "checkpoints",
        use_wandb: bool = False,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        compile_model: bool = False,
    ):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and torch.cuda.is_available()

        # Move model to device
        self.model = model.to(device)

        # Optionally compile model (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("Model compiled!")

        # Mixed precision scaler (use bfloat16 on H100 for best performance)
        # bfloat16 doesn't need loss scaling, but we use GradScaler for compatibility
        self.scaler = GradScaler(enabled=self.use_amp)

        if self.use_amp:
            print(f"Mixed precision training enabled (bfloat16)")
        if self.grad_accum_steps > 1:
            print(
                f"Gradient accumulation: {self.grad_accum_steps} steps (effective batch size: {train_loader.batch_size * self.grad_accum_steps})"
            )

        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int):
        """Train for one epoch with AMP and gradient accumulation."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulated_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device (non_blocking for async transfer)
            audio = batch["audio"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Forward pass with automatic mixed precision
            with autocast(dtype=torch.bfloat16, enabled=self.use_amp):
                outputs = self.model(
                    audio=audio,
                    labels=labels,
                )
                # Scale loss for gradient accumulation
                loss = outputs.loss / self.grad_accum_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item() * self.grad_accum_steps

            # Update weights every grad_accum_steps
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Unscale gradients and clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(
                    set_to_none=True
                )  # More efficient than zero_grad()
                self.scheduler.step()

                # Logging
                total_loss += accumulated_loss
                num_batches += 1
                self.global_step += 1

                pbar.set_postfix(
                    {
                        "loss": accumulated_loss,
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                )

                if self.use_wandb:
                    wandb.log(
                        {
                            "train/loss": accumulated_loss,
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "global_step": self.global_step,
                        }
                    )

                accumulated_loss = 0

        # Handle remaining batches if not divisible by grad_accum_steps
        if accumulated_loss > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            total_loss += accumulated_loss
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self):
        """Run validation with AMP."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            audio = batch["audio"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Use AMP for validation too (faster inference)
            with autocast(dtype=torch.bfloat16, enabled=self.use_amp):
                outputs = self.model(
                    audio=audio,
                    labels=labels,
                )

            total_loss += outputs.loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if self.use_wandb:
            wandb.log({"val/loss": avg_loss, "global_step": self.global_step})

        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.model.config,
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / "latest.pt")

        # Save best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.output_dir / "best.pt")
            print(f"New best model saved with val_loss={val_loss:.4f}")

    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            self.save_checkpoint(epoch, val_loss)


def main():
    parser = argparse.ArgumentParser(description="Train VoxLM model")

    # YAML config option (recommended)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., configs/voxlm-2b.yaml). CLI args override config values.",
    )

    # Model selection (used if --config not provided)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model variant to train (e.g., voxlm-2b, voxlm-9b-global). Ignored if --config is provided.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda, cpu, or auto"
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=None,
        help="Training data split (e.g., train-clean-100, dev-clean)",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default=None,
        help="Validation data split (e.g., dev-clean, dev-other)",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument("--wandb-project", type=str, default=None)

    # Performance optimization arguments
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile() for model optimization (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (default: 8 for GPU, 0 for CPU)",
    )

    args = parser.parse_args()

    # ==========================================================================
    # Load configuration from YAML or use CLI defaults
    # ==========================================================================
    if args.config:
        print(f"Loading configuration from: {args.config}")
        full_config = load_full_config(args.config)

        # Extract values from YAML config (CLI args override these)
        model_name = args.model or full_config.model.name
        data_dir = args.data or full_config.data.data_dir
        output_dir = args.output or full_config.output.checkpoint_dir
        epochs = (
            args.epochs
            if args.epochs is not None
            else full_config.training.phase1_epochs
        )
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else full_config.training.phase1_batch_size
        )
        lr = args.lr if args.lr is not None else full_config.training.phase1_lr
        warmup_steps = (
            args.warmup_steps
            if args.warmup_steps is not None
            else full_config.training.phase1_warmup_steps
        )
        device = args.device or full_config.device
        train_split = args.train_split or full_config.data.train_split
        val_split = args.val_split or full_config.data.val_split
        use_wandb = args.wandb or full_config.output.use_wandb
        wandb_project = args.wandb_project or full_config.output.wandb_project
        grad_accum = (
            args.grad_accum
            if args.grad_accum is not None
            else full_config.training.phase1_grad_accum
        )
        use_amp = not args.no_amp and full_config.training.use_amp
        compile_model = args.compile or full_config.training.compile_model
        num_workers = (
            args.num_workers
            if args.num_workers is not None
            else full_config.training.num_workers
        )

        # Use model config from YAML
        model_config = full_config.model
    else:
        # Use CLI arguments with defaults
        model_name = args.model or "voxlm-2b"
        data_dir = args.data or "./data"
        output_dir = args.output or "./checkpoints"
        epochs = args.epochs if args.epochs is not None else 10
        batch_size = args.batch_size if args.batch_size is not None else 8
        lr = args.lr if args.lr is not None else 1e-4
        warmup_steps = args.warmup_steps if args.warmup_steps is not None else 1000
        device = args.device or "auto"
        train_split = args.train_split or "train-clean-100"
        val_split = args.val_split or "dev-clean"
        use_wandb = args.wandb
        wandb_project = args.wandb_project or "voxlm"
        grad_accum = args.grad_accum if args.grad_accum is not None else 1
        use_amp = not args.no_amp
        compile_model = args.compile
        num_workers = args.num_workers if args.num_workers is not None else 8

        # Use pre-defined model config
        model_config = get_config(model_name)

    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup PyTorch optimizations (TF32, cuDNN benchmark, etc.)
    setup_torch_optimizations()

    # Initialize wandb
    if use_wandb:
        if not HAS_WANDB:
            print("Warning: wandb not installed. Skipping wandb logging.")
            print("Install with: pip install wandb")
            use_wandb = False
        else:
            wandb.init(
                project=wandb_project,
                config={
                    "model": model_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "grad_accum": grad_accum,
                    "train_split": train_split,
                    "val_split": val_split,
                },
                name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

    print(f"Loading model: {model_name}")
    model = VoxLM(model_config)
    model.print_trainable_parameters()

    # Verify tokenizer setup for EOS learning
    # CRITICAL: pad_token must be different from eos_token for EOS to be learned
    print(f"\nTokenizer setup:")
    print(
        f"  eos_token: {repr(model.tokenizer.eos_token)} (id={model.tokenizer.eos_token_id})"
    )
    print(
        f"  pad_token: {repr(model.tokenizer.pad_token)} (id={model.tokenizer.pad_token_id})"
    )
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        print(
            "  WARNING: pad_token == eos_token! Model will NOT learn to generate EOS!"
        )
    else:
        print("  OK: pad_token != eos_token - Model will learn to generate EOS")

    print(f"\nLoading dataset from: {data_dir}")
    print(f"  Train split: {train_split}")
    print(f"  Val split: {val_split}")
    train_dataset = LibriSpeechDataset(
        data_dir,
        split=train_split,
        max_audio_length=model_config.max_audio_length,
    )
    val_dataset = LibriSpeechDataset(
        data_dir,
        split=val_split,
        max_audio_length=model_config.max_audio_length,
    )

    # Create picklable collate function
    collate_fn = CollateFn(model.tokenizer, model_config.max_text_length)

    # Determine number of workers
    import platform

    if platform.system() == "Darwin":
        actual_num_workers = 0  # macOS has multiprocessing issues
    elif device == "cpu":
        actual_num_workers = 0
    else:
        actual_num_workers = num_workers

    # Use pin_memory for faster CPU->GPU transfer
    pin_memory = device == "cuda"

    print(f"DataLoader: {actual_num_workers} workers, pin_memory={pin_memory}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=2 if actual_num_workers > 0 else None,
        persistent_workers=actual_num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=2 if actual_num_workers > 0 else None,
        persistent_workers=actual_num_workers > 0,
    )

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    # Scheduler (account for gradient accumulation)
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"\nTraining config:")
    print(f"  Config file: {args.config or 'None (using CLI args)'}")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  AMP enabled: {use_amp}")
    print(f"  torch.compile: {compile_model}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=f"{output_dir}/{model_name}",
        use_wandb=use_wandb,
        grad_accum_steps=grad_accum,
        use_amp=use_amp,
        compile_model=compile_model,
    )

    trainer.train(epochs)

    print("Training complete!")


if __name__ == "__main__":
    main()
