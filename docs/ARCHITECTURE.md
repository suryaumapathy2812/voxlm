# VoxLM: Model Architecture (v2.1)

> **Version 2.1** - Critical fixes based on deep research of Whisper timing.py and CrisperWhisper.
> See `DEEP_RESEARCH.md` for full analysis.

## Executive Summary

**Goal:** Any Audio Encoder + Any LLM = Intelligent Transcription with Word-Level Timestamps

**Key Design Principles:**
1. **Modular**: Swap audio encoders (Whisper, IndicWhisper, MMS) and LLMs (Qwen, Llama, Phi)
2. **Timestamps**: Cross-attention + DTW alignment (not frame classification)
3. **Alignment**: Explicit cross-attention layers between text and audio
4. **Confidence**: Token probability from LLM (not separate head)

**Critical Fixes in v2.1 (from deep research):**
1. **CRITICAL**: Use PURE audio embeddings for alignment, NOT LLM hidden states
2. **HIGH**: Alignment head selection (like Whisper's `model.alignment_heads`)
3. **HIGH**: CrisperWhisper-style attention supervision loss (cosine similarity)

---

## Architecture Overview

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
│   │ Wav2Vec2    │      │ 4x down   │      │ Gemma           │   │
│   └─────────────┘      └───────────┘      └─────────────────┘   │
│                              │                                   │
│                              │ PURE audio_embeds (stored)       │
│                              ▼                                   │
│                    ┌─────────────────────┐                      │
│                    │ Alignment Module    │                      │
│                    │ (Cross-Attention)   │                      │
│                    │ Query: text_hidden  │                      │
│                    │ Key/Val: audio_emb  │                      │
│                    └─────────────────────┘                      │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────────┐                      │
│                    │ DTW + Token Prob    │                      │
│                    │ → Timestamps        │                      │
│                    │ → Confidence        │                      │
│                    └─────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Variants

### POC Models (for testing)

| Variant | Audio Encoder | LLM | Total Params | Use Case |
|---------|---------------|-----|--------------|----------|
| `voxlm-0.5b` | Whisper-tiny (39M) | Qwen2-0.5B | ~560M | Edge, mobile |
| `voxlm-2b` | Whisper-small (244M) | Qwen2-1.5B | ~1.9B | Balanced (default) |
| `voxlm-4b-multilingual` | Whisper-large-v3-turbo | Qwen2.5-3B | ~4B | Multilingual POC |

### Production Models

| Variant | Audio Encoder | LLM | Total Params | Use Case |
|---------|---------------|-----|--------------|----------|
| `voxlm-9b-global` | Whisper-large-v3 | Qwen2.5-7B | ~9B | 99 languages |
| `voxlm-9b-india` | IndicWhisper-large | Qwen2.5-7B | ~9B | 12+ Indian languages |

### Alternative LLM Variants

| Variant | Audio Encoder | LLM | Notes |
|---------|---------------|-----|-------|
| `voxlm-llama-8b` | Whisper-large-v3 | Llama-3.1-8B | Alternative LLM |
| `voxlm-phi-4` | Whisper-large-v3 | Phi-4 | Microsoft LLM |
| `voxlm-gemma-9b` | Whisper-large-v3 | Gemma-2-9B | Google LLM |

### Edge/Mobile

| Variant | Audio Encoder | LLM | Notes |
|---------|---------------|-----|-------|
| `voxlm-edge` | Whisper-tiny | Qwen2.5-0.5B | Fast inference |

---

## Component Details

### 1. Audio Encoder (Swappable)

```python
class AudioEncoder(nn.Module):
    """Wrapper for any audio encoder (Whisper, IndicWhisper, MMS, etc.)."""
    
    def __init__(self, model_name: str = "openai/whisper-small", freeze: bool = True):
        super().__init__()
        # Auto-detect encoder type from model_name
        if "whisper" in model_name.lower():
            self.encoder = WhisperModel.from_pretrained(model_name).encoder
        elif "indicwhisper" in model_name.lower():
            self.encoder = WhisperModel.from_pretrained(model_name).encoder
        elif "mms" in model_name.lower():
            self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        # ... more encoder types
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [batch, samples] @ 16kHz
        # output: [batch, frames, dim] @ 50Hz
        return self.encoder(audio).last_hidden_state
```

| Encoder | Dim | Layers | Params | Frame Rate | Languages |
|---------|-----|--------|--------|------------|-----------|
| Whisper-tiny | 384 | 4 | 39M | 50Hz | 99 |
| Whisper-small | 768 | 12 | 244M | 50Hz | 99 |
| Whisper-large-v3 | 1280 | 32 | 1.5B | 50Hz | 99 |
| IndicWhisper-large | 1280 | 32 | 1.5B | 50Hz | 12+ Indian |

### 2. Projection Layer (Adapts dimensions + downsamples)

```python
class AudioProjection(nn.Module):
    """Project audio features to LLM space with temporal downsampling."""
    
    def __init__(
        self, 
        audio_dim: int,   # From encoder (e.g., 768 for Whisper-small)
        llm_dim: int,     # To LLM (e.g., 1536 for Qwen2-1.5B)
        downsample_factor: int = 4,  # 50Hz → 12.5Hz
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Two-layer MLP projection
        self.proj = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.Dropout(dropout),
        )
        
        # Aggressive downsampling: 50Hz → 12.5Hz
        self.downsample = nn.Conv1d(
            llm_dim, llm_dim, 
            kernel_size=downsample_factor, 
            stride=downsample_factor
        )
```

**Why 4x downsampling?**
- Reduces sequence length (faster attention)
- 80ms resolution is sufficient for word boundaries
- Qwen-TTS uses 12.5Hz successfully
- Reduces memory by 4x

### 3. LLM Backbone (Swappable)

```python
# Qwen2/2.5
self.llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B")

# Llama 3
self.llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Phi-4
self.llm = AutoModelForCausalLM.from_pretrained("microsoft/phi-4")

# Apply LoRA for efficient fine-tuning
if config.use_lora:
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    self.llm = get_peft_model(self.llm, lora_config)
```

### 4. Cross-Attention Alignment Module

```python
class AlignmentModule(nn.Module):
    """
    Cross-attention module for audio-text alignment.
    
    CRITICAL: Uses PURE audio embeddings from projection layer,
    NOT LLM hidden states (which are contaminated by self-attention).
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        text_hidden: torch.Tensor,   # [batch, text_len, dim] - from LLM
        audio_embeds: torch.Tensor,  # [batch, audio_len, dim] - PURE from projection!
    ) -> torch.Tensor:
        """
        Returns:
            alignment: [batch, text_len, audio_len] - attention weights
        """
        all_weights = []
        x = text_hidden
        
        for layer in self.layers:
            x, attn_weights = layer(
                query=x,
                key=audio_embeds,
                value=audio_embeds,
                return_attention=True,
            )
            all_weights.append(attn_weights)
        
        # Average across layers and heads
        alignment = torch.stack(all_weights).mean(dim=0).mean(dim=1)
        return alignment
```

### 5. DTW Timestamp Extractor

```python
class TimestampExtractor:
    """
    Extract word-level timestamps from alignment weights using DTW.
    
    This is the same approach Whisper uses (see whisper/timing.py).
    """
    
    def __init__(
        self, 
        frame_duration_ms: float = 80.0,  # 12.5Hz
        median_filter_width: int = 7,
    ):
        self.frame_duration_ms = frame_duration_ms
        self.median_filter_width = median_filter_width
    
    def __call__(
        self,
        alignment: torch.Tensor,  # [batch, text_len, audio_frames]
        token_ids: torch.Tensor,
        tokenizer,
        audio_duration: float,
    ) -> list:
        """
        Extract word timestamps from alignment weights.
        
        1. Apply median filter for smoothness
        2. Normalize alignment weights
        3. Run DTW to find optimal monotonic path
        4. Convert path to word timestamps
        """
        # ... implementation
```

### 6. Confidence from Token Probability

```python
class ConfidenceExtractor:
    """
    Extract confidence scores from LLM token probabilities.
    
    No extra parameters needed - LLM probabilities are already calibrated.
    """
    
    @staticmethod
    def from_scores(
        scores: List[torch.Tensor],  # Generation scores
        token_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get confidence as probability of generated token."""
        all_probs = []
        for i, score in enumerate(scores):
            probs = F.softmax(score / temperature, dim=-1)
            token_prob = probs[0, token_ids[0, i]].item()
            all_probs.append(token_prob)
        return torch.tensor(all_probs)
```

---

## Training Strategy

### Phase 1: Transcription Training
- **Data**: LibriSpeech (960h)
- **Trainable**: Projection layer + LoRA
- **Loss**: Cross-entropy on transcription
- **Goal**: Audio embeddings align with LLM space

### Phase 2: Alignment Training
- **Data**: LibriSpeech with Whisper-generated timestamps
- **Trainable**: Alignment module + projection + LoRA
- **Loss**: `CE(text) + alignment_loss`
- **Goal**: Cross-attention learns to align text to audio

### Phase 3: Domain Fine-tuning (Optional)
- **Data**: Domain-specific audio (accents, noise, children)
- **Trainable**: All components
- **Goal**: Robust transcription for specific use cases

---

## Key Differences: v1 vs v2

| Aspect | v1 (Wrong) | v2 (Correct) |
|--------|------------|--------------|
| **Timestamps** | Classification over 3000 frames | Cross-attention + DTW |
| **Alignment** | Implicit in self-attention | Explicit cross-attention module |
| **Confidence** | Separate untrained head | Token probability (free) |
| **Downsampling** | 2x (50Hz → 25Hz) | 4x (50Hz → 12.5Hz) |
| **Audio source** | LLM hidden states | Pure projection output |
| **Memory** | High (3000-class logits) | Low (attention weights) |

---

## Configuration

```python
@dataclass
class VoxLMConfig:
    # Model variant name
    name: str = "voxlm-2b"
    
    # Audio Encoder
    audio_encoder: str = "openai/whisper-small"
    audio_encoder_dim: int = 768
    freeze_audio_encoder: bool = True
    
    # LLM Backbone
    llm_model: str = "Qwen/Qwen2-1.5B"
    llm_dim: int = 1536
    freeze_llm: bool = True
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    
    # Projection
    downsample_factor: int = 4  # 50Hz → 12.5Hz
    
    # Alignment
    alignment_num_layers: int = 2
    alignment_num_heads: int = 8
    alignment_loss_weight: float = 1.0
```

---

## References

1. **Whisper timing.py** - OpenAI's DTW implementation for word timestamps
   - https://github.com/openai/whisper/blob/main/whisper/timing.py

2. **CrisperWhisper** - Improved timestamps via attention supervision
   - https://arxiv.org/abs/2408.16589

3. **Qwen-TTS** - Uses 12.5Hz frame rate successfully
   - https://github.com/QwenLM/Qwen3-TTS

4. **IndicWhisper** - Indian language support
   - https://github.com/AI4Bharat/IndicWhisper
