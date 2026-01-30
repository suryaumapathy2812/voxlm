# Getting Started with VoxLM

> **Complete guide to reproduce the VoxLM training pipeline** - From setup to inference.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Download Data](#download-data)
4. [Phase 1: Transcription Training](#phase-1-transcription-training)
5. [Generate Timestamps](#generate-timestamps)
6. [Phase 2: Alignment Training](#phase-2-alignment-training)
7. [Inference](#inference)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 24GB VRAM (RTX 3090/4090) | 40GB+ (A100/H100) |
| RAM | 32GB | 64GB |
| Storage | 50GB | 100GB |

### Software Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- ffmpeg (for audio processing)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/foreverlearning/intervoo.git
cd intervoo/research/voxlm
```

### 2. Install uv (Recommended)

We use `uv` for fast, reliable Python package management:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 3. Create Virtual Environment & Install Dependencies

```bash
# Create venv and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

uv pip install -r requirements.txt
```

### 4. Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### 5. (Optional) Set up Weights & Biases

For experiment tracking:

```bash
pip install wandb
wandb login
```

---

## Download Data

We use LibriSpeech for training. The scripts will download automatically, but you can also download manually:

### Option A: Use Download Script

```bash
uv run python scripts/download_data.py --dataset librispeech --split train-clean-100
uv run python scripts/download_data.py --dataset librispeech --split dev-clean
```

### Option B: Manual Download

```bash
# Create data directory
mkdir -p data/LibriSpeech

# Download train-clean-100 (~6GB)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz -C data/

# Download dev-clean (~350MB)
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz -C data/

# Clean up
rm *.tar.gz
```

### Expected Directory Structure

```
data/
└── LibriSpeech/
    ├── train-clean-100/
    │   ├── 19/
    │   ├── 26/
    │   └── ...
    └── dev-clean/
        ├── 84/
        ├── 174/
        └── ...
```

---

## Phase 1: Transcription Training

Phase 1 teaches the model to transcribe audio to text.

### Training Command

```bash
uv run python scripts/train.py \
    --model voxlm-2b \
    --data data/LibriSpeech/train-clean-100 \
    --val-data data/LibriSpeech/dev-clean \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-4 \
    --output ./checkpoints/voxlm-2b
```

### Parameters Explained

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model config (voxlm-2b, voxlm-9b-global, etc.) | Required |
| `--data` | Training data path | Required |
| `--val-data` | Validation data path | Required |
| `--epochs` | Number of epochs | 5 |
| `--batch-size` | Batch size (reduce if OOM) | 8 |
| `--lr` | Learning rate | 1e-4 |
| `--output` | Checkpoint directory | ./checkpoints |

### Available Model Variants

| Model | Description | GPU Memory |
|-------|-------------|------------|
| `voxlm-0.5b` | Whisper-tiny + Qwen2-0.5B | ~8GB |
| `voxlm-2b` | Whisper-small + Qwen2-1.5B (default) | ~16GB |
| `voxlm-9b-global` | Whisper-large-v3 + Qwen2.5-7B | ~40GB |
| `voxlm-9b-india` | IndicWhisper + Qwen2.5-7B | ~40GB |

### Expected Output

```
Epoch 0: train_loss=2.1234, val_loss=1.8765
Epoch 1: train_loss=0.8234, val_loss=0.5432
...
Epoch 4: train_loss=0.2134, val_loss=0.2050
```

### Training Time

| GPU | Time (5 epochs) |
|-----|-----------------|
| H100 | ~50 minutes |
| A100 | ~1.5 hours |
| RTX 4090 | ~2.5 hours |
| RTX 3090 | ~4 hours |

### Checkpoint Location

```
checkpoints/voxlm-2b/
├── best.pt      # Best validation loss
└── latest.pt    # Most recent
```

---

## Generate Timestamps

Before Phase 2, we need word-level timestamps from Whisper.

### Generate for Training Data

```bash
# Single process (slower but simpler)
uv run python scripts/generate_timestamps.py \
    --data data/LibriSpeech/train-clean-100 \
    --output data/timestamps/train-clean-100

# Parallel processing (4x faster, recommended)
uv run python scripts/generate_timestamps.py \
    --data data/LibriSpeech/train-clean-100 \
    --output data/timestamps/train-clean-100 \
    --parallel 4
```

### Generate for Validation Data

```bash
uv run python scripts/generate_timestamps.py \
    --data data/LibriSpeech/dev-clean \
    --output data/timestamps/dev-clean \
    --parallel 4
```

### Parameters Explained

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | LibriSpeech data path | Required |
| `--output` | Output directory | Required |
| `--parallel` | Number of parallel workers | 1 |
| `--model` | Whisper model | turbo |
| `--max-wer` | Max WER to accept | 0.1 |

### Expected Output

```
Processing: 100%|----------------| 28539/28539 [1:15:00<00:00]
Saved 26859 samples (94.1% success rate)
Output: data/timestamps/train-clean-100/timestamps_manifest.json
```

### Timestamp Generation Time

| Dataset | Single Process | 4 Workers |
|---------|---------------|-----------|
| train-clean-100 | ~3.5 hours | ~1 hour |
| dev-clean | ~20 minutes | ~5 minutes |

---

## Phase 2: Alignment Training

Phase 2 teaches the model to produce accurate word-level timestamps.

### Training Command

```bash
uv run python scripts/train_with_timestamps.py \
    --model voxlm-2b \
    --checkpoint checkpoints/voxlm-2b/best.pt \
    --timestamps data/timestamps/train-clean-100/timestamps_manifest.json \
    --val-timestamps data/timestamps/dev-clean/timestamps_manifest.json \
    --epochs 3 \
    --batch-size 16 \
    --output ./checkpoints/voxlm-2b
```

### Parameters Explained

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--checkpoint` | Phase 1 checkpoint | Required |
| `--timestamps` | Training timestamps | Required |
| `--val-timestamps` | Validation timestamps | Required |
| `--epochs` | Number of epochs | 3 |
| `--batch-size` | Batch size | 16 |
| `--alignment-weight` | Alignment loss weight | 1.0 |

### Expected Output

```
Phase 1 model loaded!
Trainable: 229,946,624 / 1,861,374,208 (12.35%)

Epoch 0: train_loss=0.3943, val_loss=0.3715
New best model saved with val_loss=0.3715
Epoch 1: train_loss=0.2856, val_loss=0.2934
...
```

### Training Time

| GPU | Time (3 epochs) |
|-----|-----------------|
| H100 | ~35 minutes |
| A100 | ~1 hour |
| RTX 4090 | ~2 hours |

---

## Inference

### Basic Inference

```bash
uv run python scripts/inference.py \
    --checkpoint checkpoints/voxlm-2b/best.pt \
    --audio path/to/audio.wav
```

### With Custom Instruction

```bash
uv run python scripts/inference.py \
    --checkpoint checkpoints/voxlm-2b/best.pt \
    --audio path/to/audio.wav \
    --instruction "5-year-old child speaking. Indian English accent."
```

### Expected Output

```
Transcription: "but in less than five minutes the staircase groaned beneath an extraordinary weight"

Word Timestamps:
  but             0.00s - 0.16s (conf: 1.00)
  in              0.16s - 0.32s (conf: 1.00)
  less            0.32s - 0.64s (conf: 1.00)
  than            0.64s - 0.80s (conf: 1.00)
  five            0.80s - 1.04s (conf: 1.00)
  minutes         1.04s - 1.44s (conf: 1.00)
  the             1.44s - 1.52s (conf: 1.00)
  staircase       1.52s - 2.08s (conf: 0.88)
  groaned         2.08s - 2.48s (conf: 0.99)
  beneath         2.48s - 2.80s (conf: 1.00)
  an              2.80s - 2.96s (conf: 1.00)
  extraordinary   2.96s - 3.60s (conf: 0.95)
  weight          3.60s - 3.99s (conf: 1.00)
```

### Python API

```python
from src.model import VoxLM
from src.config import get_config
import torch

# Load model
config = get_config("voxlm-2b")
model = VoxLM(config)

# Load checkpoint
checkpoint = torch.load("checkpoints/voxlm-2b/best.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to("cuda").eval()

# Transcribe
result = model.transcribe(
    audio=audio_tensor,  # [samples] @ 16kHz
    instruction="Transcribe the following audio."
)

print(result["text"])
for word in result["words"]:
    print(f"  {word['word']}: {word['start']:.2f}s - {word['end']:.2f}s (conf: {word['confidence']:.2f})")
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: `CUDA out of memory` error

**Solutions**:
1. Reduce batch size: `--batch-size 4` or `--batch-size 2`
2. Use a smaller model variant: `--model voxlm-0.5b`
3. Enable gradient checkpointing (if available)

### Slow Training

**Symptom**: Training is slower than expected

**Solutions**:
1. Ensure CUDA is being used: Check for `Using device: cuda` in output
2. Enable mixed precision (enabled by default)
3. Increase number of dataloader workers

### Timestamps Beyond Audio Duration

**Symptom**: Last words have timestamps like 10s-30s for 4s audio

**Solutions**:
1. Train more epochs in Phase 2
2. This is expected behavior before alignment training converges

### Confidence Scores All 0.00

**Symptom**: All words show confidence 0.00

**Solution**: This was a bug in earlier versions. Make sure you have the latest code with the fix in `src/model.py`.

### ffmpeg Not Found

**Symptom**: `FileNotFoundError: ffmpeg not found`

**Solution**: Install ffmpeg (see Installation section)

### HuggingFace Rate Limits

**Symptom**: `Too many requests` error when downloading models

**Solutions**:
1. Set HuggingFace token: `export HF_TOKEN=your_token_here`
2. Or login: `huggingface-cli login`

---

## Quick Reference

### Full Pipeline (Copy-Paste)

```bash
# 1. Setup
git clone https://github.com/foreverlearning/intervoo.git
cd intervoo/research/voxlm
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Download data
uv run python scripts/download_data.py --dataset librispeech --split train-clean-100
uv run python scripts/download_data.py --dataset librispeech --split dev-clean

# 3. Phase 1: Transcription training
uv run python scripts/train.py \
    --model voxlm-2b \
    --data data/LibriSpeech/train-clean-100 \
    --val-data data/LibriSpeech/dev-clean \
    --epochs 5 \
    --batch-size 8 \
    --output ./checkpoints/voxlm-2b

# 4. Generate timestamps
uv run python scripts/generate_timestamps.py \
    --data data/LibriSpeech/train-clean-100 \
    --output data/timestamps/train-clean-100 \
    --parallel 4

uv run python scripts/generate_timestamps.py \
    --data data/LibriSpeech/dev-clean \
    --output data/timestamps/dev-clean \
    --parallel 4

# 5. Phase 2: Alignment training
uv run python scripts/train_with_timestamps.py \
    --model voxlm-2b \
    --checkpoint checkpoints/voxlm-2b/best.pt \
    --timestamps data/timestamps/train-clean-100/timestamps_manifest.json \
    --val-timestamps data/timestamps/dev-clean/timestamps_manifest.json \
    --epochs 3 \
    --batch-size 16 \
    --output ./checkpoints/voxlm-2b

# 6. Test inference
uv run python scripts/inference.py \
    --checkpoint checkpoints/voxlm-2b/best.pt \
    --audio data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac
```

---

## Next Steps

After completing the basic training:

1. **Benchmark**: Evaluate WER on full dev-clean set
2. **Fine-tune**: Train on domain-specific data
3. **Multilingual**: Use `voxlm-9b-global` or `voxlm-9b-india` for multilingual support
4. **Deploy**: Export model for production inference

---

## Support

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions
- **Contributing**: PRs welcome!
