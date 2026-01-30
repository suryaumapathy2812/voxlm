# VoxLM: Modular Speech-to-Text with LLM Intelligence

> **Any Audio Encoder + Any LLM = Intelligent Transcription with Word-Level Timestamps**

VoxLM is a modular ASR architecture that combines any audio encoder (Whisper, IndicWhisper, MMS) with any LLM backbone (Qwen, Llama, Phi, Gemma) to produce intelligent, context-aware transcriptions with precise word-level timestamps and calibrated confidence scores.

## Features

- **Modular architecture** - Swap audio encoders and LLMs for different use cases
- **Prompt-aware transcription** - Provide context to improve accuracy
- **Word-level timestamps** - Precise timing for every word (via cross-attention + DTW)
- **Confidence scores** - Know when to trust the output (from token probabilities)
- **Hallucination prevention** - Confidence-based stopping criteria

## Model Variants

| Variant | Audio Encoder | LLM | Use Case |
|---------|---------------|-----|----------|
| `voxlm-0.5b` | Whisper-tiny | Qwen2-0.5B | Edge, mobile |
| **`voxlm-2b`** | Whisper-small | Qwen2-1.5B | Balanced (default) |
| `voxlm-4b-multilingual` | Whisper-large-v3-turbo | Qwen2.5-3B | Multilingual POC |
| `voxlm-9b-global` | Whisper-large-v3 | Qwen2.5-7B | Production (99 languages) |
| `voxlm-9b-india` | IndicWhisper | Qwen2.5-7B | Production (Indian languages) |
| `voxlm-llama-8b` | Whisper-large-v3 | Llama-3.1-8B | Alternative LLM |
| `voxlm-phi-4` | Whisper-large-v3 | Phi-4 | Alternative LLM |
| `voxlm-edge` | Whisper-tiny | Qwen2.5-0.5B | Mobile/fast inference |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/voxlm.git
cd voxlm

# Option 1: Using uv (recommended)
uv sync

# Option 2: Using pip
pip install -e .

# Option 3: Manual pip install
pip install torch torchaudio transformers peft accelerate soundfile openai-whisper librosa tqdm pyyaml
```

### Inference

```python
from src.model import VoxLM
from src.config import get_config
import torch

# Load model
config = get_config("voxlm-2b")
model = VoxLM(config)
model.load_state_dict(torch.load("checkpoints/voxlm-2b/best.pt")["model_state_dict"])
model = model.to("cuda").eval()

# Transcribe
result = model.transcribe(
    audio=audio_tensor,  # [samples] @ 16kHz
    instruction="Clear English speech."
)

print(result["text"])
# "hello world"

print(result["words"])
# [
#   {"word": "hello", "start": 0.12, "end": 0.45, "confidence": 0.94},
#   {"word": "world", "start": 0.48, "end": 0.82, "confidence": 0.91},
# ]
```

### Command Line

```bash
# Using uv
uv run python scripts/inference.py \
    --config configs/voxlm-2b.yaml \
    --audio path/to/audio.wav

# Or with direct python
python scripts/inference.py \
    --checkpoint checkpoints/voxlm-2b/best.pt \
    --audio path/to/audio.wav
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         VoxLM                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Audio Encoder          Projection           LLM                │
│   (Swappable)            (Adapts)            (Swappable)        │
│                                                                  │
│   ┌─────────────┐      ┌───────────┐      ┌─────────────────┐   │
│   │ Whisper     │      │ audio_dim │      │ Qwen2/2.5       │   │
│   │ IndicWhisper│ ───► │    to     │ ───► │ Llama 3         │   │
│   │ MMS         │      │ llm_dim   │      │ Phi-4           │   │
│   │ Wav2Vec2    │      │           │      │ Gemma           │   │
│   └─────────────┘      └───────────┘      └─────────────────┘   │
│                                                                  │
│   + Alignment Module (for timestamps)                           │
│   + Confidence Extraction (from token probabilities)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Training

### Quick Training (Recommended)

One command to train a complete model:

```bash
# Using uv (recommended)
uv sync  # Install dependencies first
UV_RUN="uv run" ./scripts/quick_train.sh configs/voxlm-2b.yaml

# Or with system Python
./scripts/quick_train.sh                          # Default (voxlm-2b)
./scripts/quick_train.sh configs/voxlm-2b.yaml    # Specific config
./scripts/quick_train.sh configs/voxlm-9b-global.yaml  # Production model
```

This automatically:
1. Downloads LibriSpeech data
2. Generates word-level timestamps
3. Runs Phase 1 (transcription training)
4. Runs Phase 2 (timestamp alignment)
5. Saves final model to `output/models/`

### Docker Training

```bash
# Build and run
docker-compose -f docker/docker-compose.yml up

# With specific config
docker-compose run --rm voxlm configs/voxlm-9b-global.yaml
```

### Manual Training

For step-by-step control, see [docs/MANUAL_TRAINING.md](docs/MANUAL_TRAINING.md).

```bash
# Using YAML config (recommended)
uv run python scripts/train.py --config configs/voxlm-2b.yaml
uv run python scripts/generate_timestamps.py --config configs/voxlm-2b.yaml --split train-clean-100 --fast
uv run python scripts/train_with_timestamps.py --config configs/voxlm-2b.yaml --checkpoint checkpoints/voxlm-2b/best.pt

# Or with CLI arguments (legacy)
python scripts/train.py \
    --model voxlm-2b \
    --data ./data \
    --epochs 10 \
    --batch-size 8

python scripts/generate_timestamps.py \
    --data ./data \
    --split train-clean-100 \
    --output ./data/timestamps \
    --fast

python scripts/train_with_timestamps.py \
    --model voxlm-2b \
    --checkpoint checkpoints/voxlm-2b/best.pt \
    --timestamps data/timestamps/train-clean-100/timestamps_manifest.json \
    --val-timestamps data/timestamps/dev-clean/timestamps_manifest.json
```

### Configuration

All settings in one YAML file:

```yaml
# configs/voxlm-2b.yaml
model:
  name: "voxlm-2b"
  audio_encoder: "openai/whisper-small"
  llm_model: "Qwen/Qwen2-1.5B"
  use_lora: true
  lora_r: 16

training:
  phase1:
    epochs: 10
    batch_size: 8
  phase2:
    epochs: 5
    batch_size: 16

data:
  train_split: "train-clean-100"
  val_split: "dev-clean"

output:
  final_model_dir: "./output/models"
```

Create custom configs:
```bash
cp configs/custom.yaml.example configs/my-model.yaml
# Edit and run
./scripts/quick_train.sh configs/my-model.yaml
```

## Output Format

```json
{
  "text": "hello world",
  "words": [
    {"word": "hello", "start": 0.12, "end": 0.45, "confidence": 0.94},
    {"word": "world", "start": 0.48, "end": 0.82, "confidence": 0.91}
  ],
  "audio_duration": 1.5
}
```

## Documentation

See the `docs/` directory for detailed documentation:

- [MANUAL_TRAINING.md](docs/MANUAL_TRAINING.md) - Step-by-step training guide
- [GETTING_STARTED.md](docs/GETTING_STARTED.md) - Complete setup and inference guide
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Detailed architecture specification
- [JOURNEY.md](docs/JOURNEY.md) - Development story, v1 vs v2, lessons learned
- [DEEP_RESEARCH.md](docs/DEEP_RESEARCH.md) - Analysis of Whisper timing and CrisperWhisper

## Project Structure

```
voxlm/
├── configs/                        # YAML configuration files
│   ├── base.yaml                   # Base config (inherited)
│   ├── voxlm-2b.yaml               # Default model
│   ├── voxlm-9b-global.yaml        # Production model
│   └── custom.yaml.example         # Template for custom configs
├── docker/                         # Docker training setup
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── src/
│   ├── model.py                    # Main VoxLM model
│   ├── alignment.py                # Cross-attention, DTW, confidence
│   ├── audio_encoder.py            # Audio encoder wrapper
│   ├── config.py                   # Model configs + YAML loading
│   └── heads.py                    # v1 heads (deprecated)
├── scripts/
│   ├── quick_train.sh              # One-command training
│   ├── train.py                    # Phase 1 training
│   ├── train_with_timestamps.py    # Phase 2 training
│   ├── generate_timestamps.py      # Whisper timestamp generation
│   ├── inference.py                # Test inference
│   └── download_data.py            # LibriSpeech download
├── docs/
│   ├── MANUAL_TRAINING.md          # Step-by-step training guide
│   ├── GETTING_STARTED.md
│   ├── ARCHITECTURE.md
│   └── JOURNEY.md
└── README.md
```

## Key Innovations

### 1. Modular Architecture

VoxLM separates the audio encoder from the LLM, allowing you to:
- Use IndicWhisper for Indian languages
- Swap to Llama or Phi for different inference characteristics
- Mix and match for your specific use case

### 2. Cross-Attention Alignment (not frame classification)

v1 tried to classify which frame each word belongs to (3000 classes). This failed.

v2 uses cross-attention between text and audio, then DTW to find the optimal monotonic alignment path. This is how Whisper does it.

### 3. Confidence from Token Probability (no extra head)

v1 had a separate confidence head that was essentially untrained.

v2 uses the LLM's token probability directly - it's already calibrated and requires no extra parameters.

### 4. Pure Audio Embeddings for Alignment

v1 used LLM hidden states (contaminated by self-attention with text).

v2 stores pure audio embeddings from the projection layer and uses them directly for cross-attention alignment.

## Backward Compatibility

Legacy model names still work:
```python
# These are equivalent:
config = get_config("qwen-stt-2b")  # Legacy name
config = get_config("voxlm-2b")     # New name

# Legacy imports work:
from src import QwenSTT, QwenSTTConfig  # Aliases to VoxLM, VoxLMConfig
```

## Future Work

- [ ] Multilingual benchmarks (FLEURS, CommonVoice)
- [ ] Translation (Tamil audio -> English text)
- [ ] Code-switching (mixed-language speech)
- [ ] Speaker diarization
- [ ] Streaming inference
- [ ] ONNX/TensorRT export

## License

MIT

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Audio encoder and timing.py reference
- [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) - Attention supervision insights
- [Qwen2](https://github.com/QwenLM/Qwen2) - LLM backbone
- [IndicWhisper](https://github.com/AI4Bharat/IndicWhisper) - Indian language support
- [LibriSpeech](https://www.openslr.org/12) - Training data
