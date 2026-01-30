# VoxLM Manual Training Guide

This guide walks you through training VoxLM step-by-step. Use this if you want to understand each phase or customize the training process.

**For quick training**, use the one-command script instead:
```bash
./scripts/quick_train.sh configs/voxlm-2b.yaml
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Step 1: Download Data](#step-1-download-data)
4. [Step 2: Generate Timestamps](#step-2-generate-timestamps)
5. [Step 3: Phase 1 Training](#step-3-phase-1-training)
6. [Step 4: Phase 2 Training](#step-4-phase-2-training)
7. [Step 5: Export Final Model](#step-5-export-final-model)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Model | VRAM | RAM | Disk |
|-------|------|-----|------|
| voxlm-0.5b | 8GB | 16GB | 10GB |
| voxlm-2b | 16GB | 32GB | 10GB |
| voxlm-9b-global | 24GB | 64GB | 20GB |

### Software Requirements

```bash
# Python 3.10+
python --version  # Should be 3.10+

# Install dependencies
pip install -r requirements.txt
pip install pyyaml openai-whisper

# Verify CUDA (optional but recommended)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Configuration

All training settings are in YAML config files under `configs/`.

### Using Pre-defined Configs

```bash
# Development (fast, small model)
configs/voxlm-2b.yaml

# Production (large, multilingual)
configs/voxlm-9b-global.yaml
```

### Creating Custom Config

```bash
# Copy template
cp configs/custom.yaml.example configs/my-model.yaml

# Edit with your settings
nano configs/my-model.yaml
```

### Key Configuration Options

```yaml
model:
  name: "voxlm-2b"                    # Model name (for checkpoints)
  audio_encoder: "openai/whisper-small"  # Whisper variant
  llm_model: "Qwen/Qwen2-1.5B"        # LLM backbone
  use_lora: true                       # Enable LoRA (recommended)
  lora_r: 16                           # LoRA rank

training:
  phase1:
    epochs: 10                         # Transcription training
    batch_size: 8
    learning_rate: 1.0e-4
  phase2:
    epochs: 5                          # Timestamp alignment
    batch_size: 16
    learning_rate: 5.0e-5

data:
  data_dir: "./data"
  train_split: "train-clean-100"
  val_split: "dev-clean"
```

---

## Step 1: Download Data

VoxLM uses LibriSpeech for training. Download the splits you need:

### Option A: Using Download Script

```bash
python scripts/download_data.py \
    --output ./data \
    --splits train-clean-100 dev-clean
```

### Option B: Manual Download

```bash
mkdir -p data && cd data

# Training data (~6GB)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz

# Validation data (~350MB)
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz

cd ..
```

### Verify Data

```bash
ls data/LibriSpeech/
# Should show: train-clean-100  dev-clean
```

---

## Step 2: Generate Timestamps

VoxLM needs word-level timestamps for Phase 2 training. We generate these using Whisper.

### Generate Timestamps

```bash
# For training split
python scripts/generate_timestamps.py \
    --data ./data \
    --split train-clean-100 \
    --output ./data/timestamps \
    --model small \
    --fast

# For validation split
python scripts/generate_timestamps.py \
    --data ./data \
    --split dev-clean \
    --output ./data/timestamps \
    --model small \
    --fast
```

### Options

| Flag | Description |
|------|-------------|
| `--model` | Whisper model size: tiny, base, small, medium, large-v3 |
| `--fast` | Use parallel audio loading (recommended) |
| `--parallel N` | Use N parallel Whisper processes (H100: use 4-6) |
| `--max-wer 0.1` | Filter samples with WER > 10% |

### Verify Timestamps

```bash
ls data/timestamps/
# Should show: train-clean-100  dev-clean

cat data/timestamps/train-clean-100/timestamps_manifest.json | head -50
# Should show JSON with audio paths and word timestamps
```

---

## Step 3: Phase 1 Training

Phase 1 trains the model for basic transcription (audio → text).

### Run Phase 1

```bash
python scripts/train.py \
    --model voxlm-2b \
    --data ./data \
    --output ./checkpoints \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-4 \
    --train-split train-clean-100 \
    --val-split dev-clean
```

### Phase 1 Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model config name | voxlm-2b |
| `--epochs` | Training epochs | 10 |
| `--batch-size` | Batch size | 8 |
| `--lr` | Learning rate | 1e-4 |
| `--grad-accum` | Gradient accumulation steps | 1 |
| `--compile` | Use torch.compile() | false |
| `--wandb` | Enable W&B logging | false |

### Monitor Training

```bash
# Watch checkpoint directory
ls -la checkpoints/voxlm-2b/

# Check latest loss
tail -f checkpoints/voxlm-2b/training.log  # If logging enabled
```

### Phase 1 Output

```
checkpoints/voxlm-2b/
├── latest.pt      # Most recent checkpoint
└── best.pt        # Best validation loss
```

---

## Step 4: Phase 2 Training

Phase 2 trains the alignment module for word-level timestamps.

### Run Phase 2

```bash
python scripts/train_with_timestamps.py \
    --model voxlm-2b \
    --checkpoint checkpoints/voxlm-2b/best.pt \
    --timestamps data/timestamps/train-clean-100/timestamps_manifest.json \
    --val-timestamps data/timestamps/dev-clean/timestamps_manifest.json \
    --output ./checkpoints \
    --epochs 5 \
    --batch-size 16 \
    --lr 5e-5
```

### Phase 2 Options

| Flag | Description | Default |
|------|-------------|---------|
| `--checkpoint` | Phase 1 checkpoint (required) | - |
| `--timestamps` | Training timestamps manifest | - |
| `--val-timestamps` | Validation timestamps manifest | - |
| `--epochs` | Training epochs | 5 |
| `--batch-size` | Batch size | 16 |
| `--lr` | Learning rate (lower than Phase 1) | 5e-5 |
| `--alignment-weight` | Weight for alignment loss | 1.0 |

### Phase 2 Output

```
checkpoints/voxlm-2b/
├── latest.pt           # Phase 1 latest
├── best.pt             # Phase 1 best
├── latest_phase2.pt    # Phase 2 latest
└── best_phase2.pt      # Phase 2 best (FINAL MODEL)
```

---

## Step 5: Export Final Model

Copy the final model to your output directory:

```bash
# Create output directory
mkdir -p output/models/voxlm-2b

# Copy final checkpoint
cp checkpoints/voxlm-2b/best_phase2.pt output/models/voxlm-2b/model.pt

# Copy config for reference
cp configs/voxlm-2b.yaml output/models/voxlm-2b/config.yaml
```

### Test the Model

```bash
python scripts/inference.py \
    --checkpoint output/models/voxlm-2b/model.pt \
    --audio data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac
```

Expected output:
```json
{
  "text": "mister quilter is the apostle of the middle classes...",
  "words": [
    {"word": "mister", "start": 0.12, "end": 0.45, "confidence": 0.98},
    {"word": "quilter", "start": 0.45, "end": 0.89, "confidence": 0.95},
    ...
  ]
}
```

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 4

# Enable gradient accumulation
--grad-accum 4  # Effective batch = 4 * 4 = 16

# Use smaller model
--model voxlm-0.5b
```

### Slow Training

```bash
# Enable AMP (automatic mixed precision)
# Already enabled by default

# Use torch.compile (PyTorch 2.0+)
--compile

# Increase workers (Linux only)
--num-workers 8
```

### Data Not Found

```bash
# Check data structure
ls -la data/LibriSpeech/

# Should be:
# data/LibriSpeech/train-clean-100/
# data/LibriSpeech/dev-clean/

# NOT:
# data/train-clean-100/  (missing LibriSpeech folder)
```

### Timestamps Generation Fails

```bash
# Use native Whisper (slower but more compatible)
python scripts/generate_timestamps.py --use-native

# Reduce WER threshold
--max-wer 0.2  # Accept more samples

# Use smaller Whisper model
--model tiny
```

### Phase 2 Checkpoint Not Found

```bash
# Ensure Phase 1 completed successfully
ls checkpoints/voxlm-2b/best.pt

# If missing, re-run Phase 1
python scripts/train.py --model voxlm-2b ...
```

---

## Training Timeline

Approximate times on different hardware:

| Hardware | Phase 1 (10 epochs) | Timestamps | Phase 2 (5 epochs) | Total |
|----------|---------------------|------------|-------------------|-------|
| H100 80GB | 2 hours | 30 min | 1 hour | ~3.5 hours |
| A100 40GB | 4 hours | 1 hour | 2 hours | ~7 hours |
| RTX 4090 | 6 hours | 1.5 hours | 3 hours | ~10 hours |
| RTX 3090 | 8 hours | 2 hours | 4 hours | ~14 hours |

---

## Next Steps

After training:

1. **Evaluate**: Run inference on test set
2. **Export**: Convert to ONNX for deployment
3. **Fine-tune**: Train on domain-specific data
4. **Deploy**: Use with FastAPI or similar

See [GETTING_STARTED.md](GETTING_STARTED.md) for inference examples.
