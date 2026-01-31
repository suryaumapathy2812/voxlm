---
license: apache-2.0
language:
- en
library_name: transformers
tags:
- speech-to-text
- automatic-speech-recognition
- whisper
- qwen2
- timestamps
- word-level-timestamps
pipeline_tag: automatic-speech-recognition
datasets:
- librispeech_asr
metrics:
- wer
model-index:
- name: VoxLM-2B
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      name: LibriSpeech (dev-clean)
      type: librispeech_asr
      config: clean
      split: validation
    metrics:
    - type: wer
      value: 6.69
      name: WER
---

# VoxLM-2B: Modular Speech-to-Text with LLM Intelligence

VoxLM is a modular Speech-to-Text system that combines audio encoders with Large Language Models for intelligent transcription with word-level timestamps and confidence scores.

## Model Description

VoxLM-2B combines:
- **Audio Encoder**: Whisper Small (244M parameters)
- **LLM Backbone**: Qwen2-1.5B
- **Projection Layer**: Bridges audio embeddings to LLM space
- **Alignment Module**: Cross-attention for word-level timestamps

Total parameters: ~1.86B (230M trainable via LoRA)

## Key Features

- **Word-level timestamps**: Precise timing for each word
- **Confidence scores**: Per-word confidence from LLM probabilities
- **Modular architecture**: Swap encoders or LLMs easily
- **Efficient training**: LoRA adapters (12% of parameters trainable)

## Performance

| Dataset | WER | EOS Generation Rate |
|---------|-----|---------------------|
| LibriSpeech dev-clean | 6.69% | 99.9% |

## Training Details

- **Phase 1**: Basic transcription (5 epochs on train-clean-100)
- **Phase 2**: Timestamp alignment (3 epochs)
- **Hardware**: NVIDIA H100 80GB
- **Training time**: ~87 minutes total

## Usage

### Option 1: With VoxLM Repository (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/suryaumapathy2812/voxlm.git
cd voxlm

# 2. Install dependencies
pip install -e .
# Or: uv sync

# 3. Download model
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('suryaumapathy2812/voxlm-2b', 'model.pt', local_dir='./models/voxlm-2b/')"
```

```python
# 4. Run inference
import torch
import soundfile as sf
from src.model import VoxLM

# Load model
checkpoint = torch.load("models/voxlm-2b/model.pt", map_location="cuda", weights_only=False)
model = VoxLM(checkpoint["config"]).to("cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load audio (16kHz)
audio, sr = sf.read("audio.wav")
audio_tensor = torch.from_numpy(audio).float().to("cuda")

# Transcribe
with torch.no_grad():
    result = model.transcribe(audio_tensor)

print(result["text"])
# Output: "hello world"

print(result["words"])
# Output: [
#   {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.98},
#   {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.95}
# ]
```

### Option 2: With AutoModel (Coming Soon)

```python
# After HuggingFace integration is complete:
from transformers import AutoModel
import torch
import soundfile as sf

# Load model directly from HuggingFace
model = AutoModel.from_pretrained(
    "suryaumapathy2812/voxlm-2b",
    trust_remote_code=True
).to("cuda")

# Load audio and transcribe
audio, sr = sf.read("audio.wav")
result = model.generate(torch.from_numpy(audio).float().to("cuda"))

print(result["text"])
print(result["words"])
```

## Architecture

```
Audio Input (16kHz)
       │
       ▼
┌─────────────────┐
│ Whisper Encoder │  (frozen)
│   (244M params) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Projection│  (trainable)
│   + Downsampling│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Qwen2-1.5B LLM │  (LoRA adapters)
│                 │
└────────┬────────┘
         │
         ├──────────────────┐
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│  Text Output    │  │ Alignment Module│
│  (transcription)│  │ (timestamps)    │
└─────────────────┘  └─────────────────┘
```

## Limitations

- Trained on English (LibriSpeech) only
- Best performance on clean audio
- May struggle with heavy accents or background noise

## Citation

```bibtex
@misc{voxlm2026,
  title={VoxLM: Modular Speech-to-Text with LLM Intelligence},
  author={Surya Umapathy},
  year={2026},
  url={https://github.com/suryaumapathy2812/voxlm}
}
```

## License

Apache 2.0
