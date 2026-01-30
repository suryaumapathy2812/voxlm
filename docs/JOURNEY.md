# VoxLM: The Development Journey

> **From concept to working model** - A chronicle of building a modular LLM-powered Speech-to-Text system with word-level timestamps and confidence scores.

---

## Table of Contents

1. [The Vision](#the-vision)
2. [v1 Architecture: The Wrong Approach](#v1-architecture-the-wrong-approach)
3. [The Problems We Discovered](#the-problems-we-discovered)
4. [v2 Architecture: The Right Approach](#v2-architecture-the-right-approach)
5. [Training Journey](#training-journey)
6. [Bugs Fixed Along the Way](#bugs-fixed-along-the-way)
7. [The VoxLM Rebrand](#the-voxlm-rebrand)
8. [Current Status](#current-status)
9. [Lessons Learned](#lessons-learned)

---

## The Vision

**Goal**: Build a modular ASR system that combines:
- **Any audio encoder** - Whisper, IndicWhisper, MMS, etc.
- **Any LLM backbone** - Qwen, Llama, Phi, Gemma, etc.
- **Word-level timestamps** - Precise timing for every word
- **Confidence scores** - Know when to trust the output

**Key differentiator**: Prompt-aware transcription. Unlike Whisper, users can provide context:

```python
result = model.transcribe(
    audio="classroom.wav",
    instruction="5-year-old child describing their family. Indian English accent."
)
```

The LLM uses this context to disambiguate unclear words ("daddy" not "data").

---

## v1 Architecture: The Wrong Approach

### Initial Design (Qwen-STT)

```
Audio (16kHz) --> Whisper Encoder --> Projection --> LLM (Qwen2)
                                                        |
                                                        v
                                              +------------------+
                                              | TimestampHead    |  <-- WRONG!
                                              | (frame classify) |
                                              +------------------+
                                              | ConfidenceHead   |  <-- WRONG!
                                              | (separate MLP)   |
                                              +------------------+
```

### What We Tried

**TimestampHead**: Classification over audio frames
```python
class TimestampHead(nn.Module):
    def __init__(self, hidden_dim, max_frames=3000):
        self.start_proj = nn.Linear(hidden_dim, max_frames)  # 3000 classes!
        self.end_proj = nn.Linear(hidden_dim, max_frames)
```

**ConfidenceHead**: Separate MLP predicting confidence
```python
class ConfidenceHead(nn.Module):
    def __init__(self, hidden_dim):
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
```

### Why We Thought This Would Work

- Seemed intuitive: "predict which frame each word starts/ends at"
- Similar to object detection (predict bounding boxes)
- Confidence head could learn from correctness signal

---

## The Problems We Discovered

### Problem 1: Frame Classification is Fundamentally Flawed

**Why it fails:**
- Classification over 1500-3000 classes is extremely hard
- No temporal structure - treats frames as independent classes
- Doesn't use audio features - only uses LLM hidden states
- Memory explosion - `[batch, seq, 3000]` logits tensor is huge
- Model memorizes rather than learns alignment

**The insight**: Whisper doesn't do frame classification! It uses **cross-attention + DTW**.

### Problem 2: Confidence Head Had No Training Signal

```python
# We claimed this loss:
# Loss: BCE(confidence, is_correct)

# But how do you define is_correct?
# - No per-token correctness labels exist
# - The head is essentially untrained
```

**The insight**: LLM already produces calibrated probabilities! Token probability = confidence. No extra head needed.

### Problem 3: Audio Embeddings Were Contaminated

We were using LLM hidden states for alignment:
```python
# WRONG - these are mixed with text via self-attention
audio_embeds = outputs.hidden_states[:, audio_start:audio_end, :]
```

**The insight**: Need PURE audio embeddings from projection layer, not after LLM self-attention.

---

## v2 Architecture: The Right Approach

### Deep Research Phase

We studied:
1. **Whisper's timing.py** - How OpenAI extracts timestamps
2. **CrisperWhisper** - Improvements via attention supervision
3. **Qwen-TTS** - How Alibaba handles audio-text alignment

### Key Learnings

| Source | Insight |
|--------|---------|
| Whisper | Cross-attention + DTW for timestamps |
| Whisper | Only specific attention heads are good for alignment |
| CrisperWhisper | Cosine similarity loss on attention weights |
| CrisperWhisper | Linear interpolation at word boundaries |
| Qwen-TTS | 12.5Hz frame rate is sufficient |

### v2 Architecture

```
Audio (16kHz) --> Whisper Encoder --> Projection (4x downsample) --> audio_embeds
                                                                        |
                                                                        | (STORED - PURE!)
                                                                        v
[instruction] [audio_embeds] [transcribe] --> LLM --> text_hidden
                                                          |
                                                          v
                                              +------------------------+
                                              | Alignment Module       |
                                              | (Cross-Attention)      |
                                              |                        |
                                              | Query: text_hidden     |
                                              | Key/Val: audio_embeds  |
                                              +------------------------+
                                                          |
                                                          v
                                              +------------------------+
                                              | DTW Timestamp Extract  |
                                              | + Token Probability    |
                                              | (confidence)           |
                                              +------------------------+
```

### Key Changes

| Aspect | v1 (Wrong) | v2 (Correct) |
|--------|------------|--------------|
| **Timestamps** | Classification over 3000 frames | Cross-attention + DTW |
| **Alignment** | Implicit in self-attention | Explicit cross-attention module |
| **Confidence** | Separate untrained head | Token probability (free!) |
| **Downsampling** | 2x (50Hz -> 25Hz) | 4x (50Hz -> 12.5Hz) |
| **Audio source** | LLM hidden states | Pure projection output |

---

## Training Journey

### Phase 1: Transcription Training

**Goal**: Teach model to transcribe audio to text

**Data**: LibriSpeech train-clean-100 (28,539 samples)

**What we trained**:
- Projection layer (audio -> LLM space)
- LoRA adapters on Qwen2

**Results**:
- 5 epochs, ~50 minutes on H100
- Final val_loss: 0.205
- Model transcribes correctly!

### Phase 1.5: EOS Token Training

**Problem discovered**: Model didn't know when to stop generating

**Solution**: Added EOS token to training targets

**Results**:
- Model now generates EOS and stops
- But still hallucinated (repeated words)

### Hallucination Fix: Confidence-Based Stopping

**Problem**: Model generated text beyond audio content

**Root cause**: Model not generating EOS token reliably

**Solution**: `ConfidenceStoppingCriteria`
```python
class ConfidenceStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        # Stop when token confidence drops below threshold
        probs = F.softmax(scores[-1], dim=-1)
        confidence = probs.max(dim=-1).values
        if confidence < self.threshold:
            self.stop_at_token = len(scores)
            return True
        return False
```

**Results**: 
- Test: "GO DO YOU HEAR" -> "go do you hear" (perfect!)
- No more hallucination

### Phase 2: Alignment Training

**Goal**: Teach model to produce accurate timestamps

**Data**: LibriSpeech with Whisper-generated timestamps
- train-clean-100: 26,861 samples (94.1% success rate)
- dev-clean: 2,410 samples (89.2% success rate)

**What we trained**:
- Alignment module (cross-attention layers)
- Continued training projection + LoRA

**Loss function**: Cross-entropy + Alignment loss (cosine similarity)

**Bug discovered**: `audio_mask` convention was inverted!
- `audio_mask` from prepare_inputs: True = valid frame
- CrossAttentionBlock expects: True = masked (ignored)
- Fix: `audio_mask = ~audio_mask`

**Results after training**:
- Timestamps mostly correct for early words
- Last 2 words had timestamps extending beyond audio duration
- Fixed by truncating alignment to actual audio frames

---

## Bugs Fixed Along the Way

### Bug 1: Loss Was ~0.0001 (Way Too Low!)

**Symptom**: Training showed suspiciously low loss

**Root cause**: When using `inputs_embeds`, HuggingFace doesn't auto-compute loss

**Fix**: Manually compute cross-entropy on text portion
```python
text_logits = self.llm.lm_head(text_hidden)
loss = F.cross_entropy(text_logits.view(-1, vocab_size), labels.view(-1))
```

### Bug 2: Confidence Scores Were 0.00

**Symptom**: All words showed confidence 0.00

**Root cause**: Wrong slicing of generated sequences
```python
# WRONG - sequences already contains only generated tokens when using inputs_embeds
new_token_ids = generated.sequences[0, prompt_len:]  # Empty!

# CORRECT
new_token_ids = generated_ids  # Already the generated tokens
```

**Fix**: Use `generated_ids` directly (already processed)

### Bug 3: Audio Mask Inverted

**Symptom**: NaN loss during alignment training

**Root cause**: Mask convention mismatch
- Our code: True = valid
- Attention: True = masked (ignored)

**Fix**: Invert mask before passing to attention
```python
if audio_mask is not None:
    audio_mask = ~audio_mask
```

### Bug 4: Timestamps Beyond Audio Duration

**Symptom**: Last words had timestamps like 10s-30s for 4s audio

**Root cause**: Alignment included padded frames (Whisper pads to 30s)

**Fix**: Truncate alignment to actual audio frames before DTW
```python
actual_audio_frames = int(audio_duration * self.config.effective_frame_rate)
alignment_truncated = alignment[:, :, :actual_audio_frames]
```

### Bug 5: Word-Level vs Token-Level Timestamps

**Symptom**: Timestamp count didn't match label count

**Root cause**: Whisper generates word-level timestamps, but model operates on subword tokens

**Fix**: Expand word timestamps to token timestamps in collate function
```python
def _expand_word_timestamps_to_tokens(self, text, word_timestamps, token_ids):
    # Tokenize each word and assign same timestamp to all its tokens
    ...
```

---

## The VoxLM Rebrand

### Why Rename?

The original name "Qwen-STT" implied:
- Only works with Qwen LLMs
- Tied to a specific architecture

But the architecture is **modular**:
- Any audio encoder (Whisper, IndicWhisper, MMS)
- Any LLM (Qwen, Llama, Phi, Gemma)

### New Name: VoxLM

**Vox** (Latin for "voice") + **LM** (Language Model) = **VoxLM**

### New Model Variants

| Variant | Use Case |
|---------|----------|
| `voxlm-0.5b` | Edge/mobile |
| `voxlm-2b` | Balanced (default) |
| `voxlm-9b-global` | Production (99 languages) |
| `voxlm-9b-india` | Indian languages |
| `voxlm-llama-8b` | Alternative LLM |
| `voxlm-phi-4` | Alternative LLM |
| `voxlm-edge` | Fast inference |

### Backward Compatibility

Legacy names still work:
```python
# These are equivalent:
config = get_config("qwen-stt-2b")  # Legacy
config = get_config("voxlm-2b")     # New

# Legacy imports work:
from src import QwenSTT  # Alias to VoxLM
```

---

## Current Status

### What Works

| Feature | Status | Notes |
|---------|--------|-------|
| Transcription | Working | Perfect accuracy on test samples |
| Hallucination prevention | Working | Confidence-based stopping |
| Confidence scores | Working | 0.88-1.00 range, correlates with correctness |
| Word timestamps | Working | Fixed with alignment truncation |
| Modular architecture | Working | Swap encoders/LLMs via config |

### Sample Output

```
Audio: "BUT IN LESS THAN FIVE MINUTES THE STAIRCASE GROANED BENEATH AN EXTRAORDINARY WEIGHT"
Duration: 3.99s

Word Timestamps:
  but             0.00s - 0.16s (conf: 1.00)
  in              0.16s - 0.32s (conf: 1.00)
  less            0.32s - 0.64s (conf: 1.00)
  than            0.64s - 0.80s (conf: 1.00)
  five            0.80s - 1.04s (conf: 1.00)
  minutes         1.04s - 1.44s (conf: 1.00)
  the             1.44s - 1.52s (conf: 1.00)
  staircase       1.52s - 2.08s (conf: 0.88)  <-- Lower confidence!
  groaned         2.08s - 2.48s (conf: 0.99)
  beneath         2.48s - 2.80s (conf: 1.00)
  an              2.80s - 2.96s (conf: 1.00)
  extraordinary   2.96s - 3.60s (conf: 0.95)
  weight          3.60s - 3.99s (conf: 1.00)
```

---

## Lessons Learned

### 1. Study Existing Solutions First

We wasted time on frame classification before studying how Whisper actually does timestamps. **Always read the source code of working systems.**

### 2. Simple Solutions Often Work Better

- Confidence: Token probability (free!) vs trained head (complex, untrained)
- Timestamps: Cross-attention + DTW (proven) vs frame classification (novel, failed)

### 3. Debug with Print Statements

Adding debug output helped us find:
- Confidence extraction bug (wrong tensor slicing)
- Mask inversion bug (NaN loss)
- Timestamp expansion bug (word vs token mismatch)

### 4. Incremental Training Works

- Phase 1: Learn to transcribe
- Phase 1.5: Learn to stop (EOS)
- Phase 2: Learn timestamps

Each phase builds on the previous, making debugging easier.

### 5. Data Quality Matters

Timestamp generation with Whisper Turbo:
- 94% success rate on train-clean-100
- 6% filtered due to WER > 10%

Filtering bad alignments prevents training on noisy labels.

### 6. Modularity Enables Flexibility

By separating audio encoder from LLM:
- Can use IndicWhisper for Indian languages
- Can swap to Llama or Phi for different inference characteristics
- Can mix and match for specific use cases

---

## Future Directions

### Immediate

1. Benchmark on full dev-clean set
2. Evaluate timestamp accuracy vs Whisper ground truth
3. Test with different audio encoders (IndicWhisper, MMS)

### Short-term

1. Add post-processing to smooth timestamps
2. Implement alignment head selection (like Whisper)
3. Add audio augmentation for robustness

### Long-term

1. **Multilingual benchmarks**: FLEURS, CommonVoice
2. **Translation**: Tamil audio -> English text
3. **Code-switching**: Handle mixed-language speech
4. **Speaker diarization**: Who said what
5. **Streaming inference**: Real-time transcription

---

## Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | Main VoxLM model |
| `src/alignment.py` | Cross-attention, DTW, confidence extraction |
| `src/config.py` | Model configurations |
| `scripts/train.py` | Phase 1 training |
| `scripts/train_with_timestamps.py` | Phase 2 training |
| `scripts/generate_timestamps.py` | Whisper timestamp generation |
| `scripts/inference.py` | Test inference |

---

## Timeline

| Date | Milestone |
|------|-----------|
| Day 1 | Initial v1 architecture, discovered frame classification issues |
| Day 2 | Deep research into Whisper timing.py, CrisperWhisper |
| Day 3 | v2 architecture design, implemented alignment module |
| Day 4 | Phase 1 training complete, discovered loss bug |
| Day 5 | Fixed loss bug, Phase 1.5 with EOS token |
| Day 6 | Hallucination fix, confidence-based stopping |
| Day 7 | Timestamp generation, Phase 2 training started |
| Day 8 | Fixed confidence extraction bug, audio mask bug |
| Day 9 | Fixed timestamp alignment bug, Phase 2 training complete |
| Day 10 | VoxLM rebrand, modular architecture |

---

## Acknowledgments

- **OpenAI Whisper** - For the timing.py reference implementation
- **CrisperWhisper** - For attention supervision insights
- **Qwen Team** - For Qwen2 LLM and Qwen-TTS architecture ideas
- **AI4Bharat** - For IndicWhisper
- **LibriSpeech** - For high-quality training data
