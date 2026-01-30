# Deep Research: Whisper Timestamps & CrisperWhisper Improvements

> **Purpose**: Complete technical analysis of Whisper's timestamp mechanism and CrisperWhisper's improvements to inform correct Qwen-STT v2 architecture.

---

## Table of Contents

1. [Whisper's Timestamp Mechanism (timing.py)](#1-whispers-timestamp-mechanism)
2. [CrisperWhisper's Key Innovations](#2-crisperwhispers-key-innovations)
3. [Gap Analysis: Our v2 vs Correct Approach](#3-gap-analysis)
4. [Correct Architecture Specification](#4-correct-architecture-specification)
5. [Implementation Checklist](#5-implementation-checklist)

---

## 1. Whisper's Timestamp Mechanism

### 1.1 Architecture Overview

Whisper uses a standard **encoder-decoder Transformer** architecture:

```
Audio (mel spectrogram) → Encoder → encoder_outputs [batch, 1500, dim]
                                          ↓
                              Decoder (with cross-attention)
                                          ↓
                              Text tokens + timestamps
```

**Critical insight**: The decoder has **cross-attention layers** that attend from text tokens to encoder outputs. These cross-attention weights are the key to timestamp extraction.

### 1.2 Cross-Attention in Whisper Decoder

From `whisper/model.py`:

```python
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        self.attn = MultiHeadAttention(n_state, n_head)  # Self-attention
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        # ...

    def forward(self, x, xa=None, ...):
        x = x + self.attn(self.attn_ln(x), ...)  # Self-attention on text
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, ...)  # Cross-attention to audio
        # ...
```

**Key points**:
- `x` = text hidden states (queries)
- `xa` = encoder outputs (keys/values) - **RAW AUDIO FEATURES**
- Cross-attention produces weights `[batch, heads, text_len, audio_len]`

### 1.3 Alignment Head Selection

Whisper doesn't use ALL cross-attention heads. It uses a **pre-selected subset** called `alignment_heads`:

```python
class Whisper(nn.Module):
    def __init__(self, dims):
        # Default: use last half of decoder layers
        all_heads = torch.zeros(dims.n_text_layer, dims.n_text_head, dtype=torch.bool)
        all_heads[dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        # Load pre-computed alignment heads from checkpoint
        array = np.frombuffer(gzip.decompress(base64.b85decode(dump)), dtype=bool)
        mask = torch.from_numpy(array).reshape(dims.n_text_layer, dims.n_text_head)
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)
```

**Why this matters**: Not all attention heads are good for alignment. Some heads learn other patterns (syntax, semantics). Only specific (layer, head) pairs produce clean alignment patterns.

### 1.4 The `find_alignment` Function (timing.py)

This is the core timestamp extraction logic:

```python
def find_alignment(model, tokenizer, text_tokens, mel, num_frames, *, medfilt_width=7, qk_scale=1.0):
    # 1. Prepare tokens with special tokens
    tokens = torch.tensor([
        *tokenizer.sot_sequence,
        tokenizer.no_timestamps,
        *text_tokens,
        tokenizer.eot,
    ])

    # 2. Install hooks to capture cross-attention weights
    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1][0])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    # 3. Run forward pass (with SDPA disabled to get attention weights)
    with torch.no_grad(), disable_sdpa():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]
        # Get token probabilities for confidence
        sampled_logits = logits[len(tokenizer.sot_sequence):, :tokenizer.eot]
        token_probs = sampled_logits.softmax(dim=-1)
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens]

    # 4. Remove hooks
    for hook in hooks:
        hook.remove()

    # 5. SELECT SPECIFIC ALIGNMENT HEADS (not all!)
    weights = torch.stack([QKs[_l][_h] for _l, _h in model.alignment_heads.indices().T])
    weights = weights[:, :, :num_frames // 2]  # Crop to actual audio length

    # 6. Apply softmax and normalize
    weights = (weights * qk_scale).softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std

    # 7. Apply median filter for smoothness
    weights = median_filter(weights, medfilt_width)

    # 8. Average across selected heads
    matrix = weights.mean(axis=0)
    matrix = matrix[len(tokenizer.sot_sequence):-1]  # Remove special tokens

    # 9. Run DTW on NEGATED matrix (similarity → cost)
    text_indices, time_indices = dtw(-matrix)

    # 10. Convert to word timestamps
    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]

    # 11. Get word probabilities (confidence)
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    return [WordTiming(word, tokens, start, end, probability) for ...]
```

### 1.5 DTW Algorithm

```python
@numba.jit(nopython=True)
def dtw_cpu(x: np.ndarray):
    """
    x: cost matrix [text_len, audio_frames] - LOWER = BETTER MATCH
    """
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)
    cost[0, 0] = 0

    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]  # Diagonal (match)
            c1 = cost[i - 1, j]      # Vertical (skip audio frame)
            c2 = cost[i, j - 1]      # Horizontal (skip text token)

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)
```

**Key insight**: DTW finds the optimal **monotonic** alignment path through the cost matrix.

### 1.6 Frame Rate and Time Precision

```python
# From whisper/audio.py
SAMPLE_RATE = 16000
HOP_LENGTH = 160  # 10ms hop
N_FFT = 400       # 25ms window

# Encoder output: 1500 frames for 30s audio
# But conv2 has stride=2, so effective rate = 50Hz / 2 = 25Hz for encoder output
# Actually: 30s * 16000 / 160 / 2 = 1500 frames

TOKENS_PER_SECOND = SAMPLE_RATE / HOP_LENGTH / 2  # = 50 tokens/sec
# But timing.py uses num_frames // 2, so effective = 25Hz

time_precision = 0.02  # 20ms per frame (50Hz)
```

---

## 2. CrisperWhisper's Key Innovations

### 2.1 Tokenizer Adjustment (Paper's Main Contribution)

**Problem**: Standard Whisper tokenizer can produce tokens that span both a word AND a pause:
```
"hello " → single token (includes trailing space)
```

This makes it impossible to accurately timestamp the pause separately from the word.

**Solution**: Retokenize so each token is EITHER a word part OR a pause/space:
```
"hello " → ["hello", " "]  # Two separate tokens
```

**Implementation**: They modified the tokenizer to split on word boundaries, ensuring:
- Word tokens never include leading/trailing spaces
- Spaces/pauses are separate tokens
- Each token can be cleanly aligned to either speech or silence

### 2.2 Attention Loss (Not in Paper - Key Innovation!)

From CrisperWhisper README:

> **New Feature**: Not mentioned in the paper is an added AttentionLoss to further improve timestamp accuracy. By specifically adding a loss to train the attention scores used for the DTW alignment using timestamped data we significantly boosted the alignment performance.

**How it works**:

1. **Ground Truth Cross-Attention**:
   - L2-normalized vector where:
     - `1` = word is active (according to forced alignment timestamps)
     - `0` = word is not active
   - Linear interpolation (4 steps = 8ms) at boundaries for smoothness

2. **Loss Function**:
   ```python
   loss = 1 - cosine_similarity(predicted_attention, ground_truth_attention)
   ```

3. **Averaged across**:
   - All predicted tokens
   - All selected alignment heads

### 2.3 Alignment Head Selection

From README:
> To choose the heads for alignment we evaluated the alignment performance of each individual decoder attention head on the timestamped TIMIT dataset. We choose the 15 best performing heads and finetune them using our attention loss.

**Process**:
1. Evaluate each (layer, head) pair on TIMIT (has precise phoneme timestamps)
2. Rank by alignment accuracy (F1 score, IOU)
3. Select top 15 heads
4. Fine-tune only these heads with attention loss

### 2.4 Training Data Preparation

1. **Timestamped datasets**: AMI IHM, TIMIT (have word-level timestamps)
2. **Generated timestamps**: Used PyTorch CTC aligner on CommonVoice
3. **Pause correction**: Applied same pause-splitting method to fix CTC aligner's overestimated pause durations

### 2.5 Audio Augmentation

From README:
> We use WavLM augmentations during Training adding random speech samples or noise to the audio wave to generally increase robustness of the transcription and stability of the alignment heads.

**Augmentation types**:
- Random speech samples (background speakers)
- Noise addition
- 1% pure noise samples (model must return empty prediction)

### 2.6 Training Tricks

1. **Audio shifting**: 50% probability to shift audio + timestamps to prevent overfitting to early positions
2. **Silence handling**: If >40ms silence before/after shifting, prepend space token so model learns to predict start time
3. **Attention clipping**: Clip predicted attention values to 0 outside ±4 seconds of ground truth word
4. **Three-stage training**:
   - Stage 1: 10,000 hours to adjust to new tokenizer
   - Stage 2: High-quality verbatim datasets only
   - Stage 3: Verbatim mixture + attention loss (6000 steps)

---

## 3. Gap Analysis: Our v2 vs Correct Approach

### Gap 1: Wrong Audio Representations (CRITICAL)

| Aspect | Our v2 (Wrong) | Correct Approach |
|--------|----------------|------------------|
| **Audio source** | LLM hidden states after self-attention | Raw encoder outputs |
| **Why wrong** | Self-attention mixes audio with text, corrupting alignment signal | Encoder outputs are pure audio features |
| **Fix** | Store raw audio embeddings separately, use for cross-attention K/V | Use `audio_embeds` from projection layer directly |

**Our wrong code**:
```python
# We were doing this:
audio_embeds = outputs.hidden_states[:, audio_start:audio_end, :]
# This is WRONG - these are mixed with text via self-attention
```

**Correct approach**:
```python
# Store audio embeddings BEFORE they go through LLM
audio_embeds = self.audio_projection(self.audio_encoder(audio))
# Use these directly for cross-attention keys/values
alignment = self.alignment_module(text_hidden, audio_embeds)  # audio_embeds is pure
```

### Gap 2: Missing Alignment Head Selection (HIGH)

| Aspect | Our v2 (Wrong) | Correct Approach |
|--------|----------------|------------------|
| **Head selection** | Average ALL heads | Select specific (layer, head) pairs |
| **Why wrong** | Many heads learn non-alignment patterns | Only ~15 heads are good for alignment |
| **Fix** | Evaluate heads on TIMIT, select best 15 | Store `alignment_heads` mask |

**Our wrong code**:
```python
# We average all heads:
alignment = torch.stack(all_weights).mean(dim=0).mean(dim=1)
```

**Correct approach**:
```python
# Select specific heads:
weights = torch.stack([
    cross_attn_weights[layer][head] 
    for layer, head in self.alignment_heads
])
alignment = weights.mean(dim=0)  # Average only selected heads
```

### Gap 3: Missing Attention Supervision Loss (HIGH)

| Aspect | Our v2 (Wrong) | Correct Approach |
|--------|----------------|------------------|
| **Alignment training** | No direct supervision | Cosine similarity loss on attention |
| **Why wrong** | Cross-attention learns implicitly, may not align well | Direct supervision forces alignment |
| **Fix** | Add attention loss during training | Implement CrisperWhisper's attention loss |

**Correct implementation**:
```python
def attention_loss(predicted_attn, word_timestamps, audio_frames, frame_rate=50.0):
    """
    predicted_attn: [batch, text_len, audio_frames] - from alignment module
    word_timestamps: List of (start_sec, end_sec) for each token
    """
    batch_size, text_len, num_frames = predicted_attn.shape
    
    # Create ground truth attention
    gt_attn = torch.zeros_like(predicted_attn)
    
    for b in range(batch_size):
        for t, (start, end) in enumerate(word_timestamps[b]):
            start_frame = int(start * frame_rate)
            end_frame = int(end * frame_rate)
            
            # Set active region to 1
            gt_attn[b, t, start_frame:end_frame] = 1.0
            
            # Linear interpolation at boundaries (4 frames = 8ms at 50Hz)
            interp_frames = 4
            for i in range(interp_frames):
                alpha = (i + 1) / (interp_frames + 1)
                if start_frame - interp_frames + i >= 0:
                    gt_attn[b, t, start_frame - interp_frames + i] = alpha
                if end_frame + i < num_frames:
                    gt_attn[b, t, end_frame + i] = 1 - alpha
    
    # L2 normalize ground truth
    gt_attn = F.normalize(gt_attn, p=2, dim=-1)
    
    # Cosine similarity loss
    cos_sim = F.cosine_similarity(predicted_attn, gt_attn, dim=-1)
    loss = 1 - cos_sim.mean()
    
    return loss
```

### Gap 4: Missing Audio Augmentation (MEDIUM)

| Aspect | Our v2 | Correct Approach |
|--------|--------|------------------|
| **Augmentation** | None | WavLM-style augmentation |
| **Why important** | Model may overfit to clean audio | Robustness to noise, background speech |
| **Fix** | Add augmentation during training | Implement noise/speech mixing |

### Gap 5: Tokenizer Doesn't Handle Pauses (MEDIUM)

| Aspect | Our v2 | Correct Approach |
|--------|--------|------------------|
| **Tokenization** | Standard Qwen tokenizer | Modified tokenizer |
| **Why important** | Tokens may span word + pause | Each token = word OR pause |
| **Fix** | Post-process tokenization | Split on word boundaries |

**Note**: For Qwen-STT, we may not need this if we're not doing verbatim transcription with pause detection. But for accurate timestamps, it helps.

---

## 4. Correct Architecture Specification

### 4.1 Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Qwen-STT v2 (Corrected)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio (16kHz) ──→ Whisper Encoder ──→ Projection ──→ audio_embeds         │
│                                                           │                 │
│                                                           │ (STORED)        │
│                                                           ↓                 │
│  [instruction] [audio_embeds] [transcribe] ──→ LLM ──→ text_hidden         │
│                                                           │                 │
│                                                           ↓                 │
│                                              ┌────────────────────────┐     │
│                                              │   Alignment Module     │     │
│                                              │                        │     │
│                                              │  Query: text_hidden    │     │
│                                              │  Key/Val: audio_embeds │←────┤
│                                              │         (PURE!)        │     │
│                                              │                        │     │
│                                              │  Select alignment_heads│     │
│                                              │  (15 best heads)       │     │
│                                              └───────────┬────────────┘     │
│                                                          │                  │
│                                                          ↓                  │
│                                              ┌────────────────────────┐     │
│                                              │   DTW + Timestamps     │     │
│                                              │                        │     │
│                                              │  1. Median filter      │     │
│                                              │  2. Normalize          │     │
│                                              │  3. DTW alignment      │     │
│                                              │  4. Extract timestamps │     │
│                                              └───────────┬────────────┘     │
│                                                          │                  │
│                                                          ↓                  │
│                                              Word timestamps + confidence   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Components

#### 4.2.1 Audio Embedding Storage

```python
class QwenSTTv2(nn.Module):
    def forward(self, audio, ...):
        # 1. Encode audio
        audio_features = self.audio_encoder(audio)  # [batch, 1500, whisper_dim]
        
        # 2. Project to LLM space
        audio_embeds = self.audio_projection(audio_features)  # [batch, 375, llm_dim]
        
        # 3. STORE audio_embeds for later use in alignment
        # This is the KEY difference - we keep pure audio embeddings
        self._cached_audio_embeds = audio_embeds
        
        # 4. Build input sequence and run through LLM
        # ... (audio_embeds goes into LLM input)
        
        # 5. Get text hidden states from LLM output
        text_hidden = outputs.hidden_states[-1][:, text_start:, :]
        
        # 6. Compute alignment using PURE audio_embeds (not LLM hidden states!)
        alignment = self.alignment_module(
            text_hidden,           # Queries from LLM
            self._cached_audio_embeds  # Keys/Values from PURE audio
        )
```

#### 4.2.2 Alignment Module with Head Selection

```python
class AlignmentModule(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Alignment head selection (to be set after evaluation)
        # Format: list of (layer_idx, head_idx) tuples
        self.alignment_heads = None  # Set via set_alignment_heads()
    
    def set_alignment_heads(self, heads: List[Tuple[int, int]]):
        """Set which (layer, head) pairs to use for alignment."""
        self.alignment_heads = heads
    
    def forward(self, text_hidden, audio_embeds):
        # Run through cross-attention layers, collecting weights
        all_weights = []
        x = text_hidden
        
        for layer_idx, layer in enumerate(self.layers):
            x, attn_weights = layer(x, audio_embeds, audio_embeds, return_attention=True)
            all_weights.append(attn_weights)  # [batch, heads, text_len, audio_len]
        
        # Select specific alignment heads
        if self.alignment_heads is not None:
            selected = []
            for layer_idx, head_idx in self.alignment_heads:
                if layer_idx < len(all_weights):
                    selected.append(all_weights[layer_idx][:, head_idx, :, :])
            weights = torch.stack(selected, dim=1)  # [batch, num_selected, text, audio]
        else:
            # Fallback: use all heads from last layer
            weights = all_weights[-1]
        
        # Average across selected heads
        alignment = weights.mean(dim=1)  # [batch, text_len, audio_len]
        
        return alignment
```

#### 4.2.3 Attention Supervision Loss

```python
class AttentionLoss(nn.Module):
    """CrisperWhisper-style attention supervision loss."""
    
    def __init__(self, frame_rate=12.5, interp_frames=4):
        super().__init__()
        self.frame_rate = frame_rate
        self.interp_frames = interp_frames
    
    def forward(
        self,
        predicted_attn: torch.Tensor,  # [batch, text_len, audio_frames]
        token_timestamps: List[List[Tuple[float, float]]],  # Per-token (start, end)
        attention_mask: torch.Tensor = None,  # [batch, text_len] - which tokens are real
    ):
        batch_size, text_len, num_frames = predicted_attn.shape
        device = predicted_attn.device
        
        # Create ground truth attention
        gt_attn = torch.zeros_like(predicted_attn)
        
        for b in range(batch_size):
            for t, (start_sec, end_sec) in enumerate(token_timestamps[b]):
                if t >= text_len:
                    break
                    
                start_frame = int(start_sec * self.frame_rate)
                end_frame = int(end_sec * self.frame_rate)
                
                # Clamp to valid range
                start_frame = max(0, min(start_frame, num_frames - 1))
                end_frame = max(start_frame + 1, min(end_frame, num_frames))
                
                # Set active region to 1
                gt_attn[b, t, start_frame:end_frame] = 1.0
                
                # Linear interpolation at boundaries
                for i in range(self.interp_frames):
                    alpha = (i + 1) / (self.interp_frames + 1)
                    
                    # Ramp up before start
                    idx = start_frame - self.interp_frames + i
                    if 0 <= idx < num_frames:
                        gt_attn[b, t, idx] = alpha
                    
                    # Ramp down after end
                    idx = end_frame + i
                    if 0 <= idx < num_frames:
                        gt_attn[b, t, idx] = 1 - alpha
        
        # L2 normalize ground truth
        gt_attn = F.normalize(gt_attn, p=2, dim=-1)
        
        # L2 normalize predictions
        pred_attn = F.normalize(predicted_attn, p=2, dim=-1)
        
        # Cosine similarity loss: 1 - cos_sim
        cos_sim = (pred_attn * gt_attn).sum(dim=-1)  # [batch, text_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            cos_sim = cos_sim * attention_mask
            loss = 1 - cos_sim.sum() / attention_mask.sum()
        else:
            loss = 1 - cos_sim.mean()
        
        return loss
```

#### 4.2.4 Training Loss

```python
def compute_loss(self, batch):
    """Combined loss for training."""
    audio = batch['audio']
    labels = batch['labels']
    token_timestamps = batch.get('token_timestamps')  # Optional
    
    # Forward pass
    outputs = self.forward(audio, labels=labels, return_alignment=True)
    
    # 1. Transcription loss (cross-entropy)
    ce_loss = outputs['loss']
    
    # 2. Attention loss (if timestamps available)
    attn_loss = 0.0
    if token_timestamps is not None and outputs['alignment'] is not None:
        attn_loss = self.attention_loss(
            outputs['alignment'],
            token_timestamps,
            attention_mask=batch.get('attention_mask')
        )
    
    # Combined loss
    total_loss = ce_loss + self.config.attention_loss_weight * attn_loss
    
    return {
        'loss': total_loss,
        'ce_loss': ce_loss,
        'attn_loss': attn_loss,
    }
```

### 4.3 Training Strategy

#### Phase 1: Audio-LLM Alignment (No timestamps needed)
- **Data**: LibriSpeech (960h)
- **Trainable**: Projection + LoRA
- **Loss**: Cross-entropy only
- **Goal**: Audio embeddings align with LLM space

#### Phase 2: Alignment Module Training (Timestamps needed)
- **Data**: LibriSpeech + forced alignment timestamps (from MFA or WhisperX)
- **Trainable**: Alignment module + projection + LoRA
- **Loss**: CE + Attention loss
- **Goal**: Cross-attention learns to align

#### Phase 3: Head Selection
- **Data**: TIMIT (precise phoneme timestamps)
- **Process**: Evaluate each (layer, head) pair, select top 15
- **Output**: `alignment_heads` configuration

#### Phase 4: Fine-tuning with Attention Loss
- **Data**: Verbatim datasets (AMI, TIMIT) + generated timestamps (CommonVoice)
- **Trainable**: Alignment heads only (freeze others)
- **Loss**: CE + Attention loss
- **Augmentation**: WavLM-style noise/speech mixing

---

## 5. Implementation Checklist

### Critical Fixes (Must Do)

- [ ] **Fix audio embedding source**: Use `audio_embeds` from projection, NOT LLM hidden states
- [ ] **Add alignment head selection**: Implement `alignment_heads` mask and selection logic
- [ ] **Implement attention loss**: CrisperWhisper-style cosine similarity loss

### High Priority

- [ ] **Prepare timestamped data**: Run MFA or WhisperX on LibriSpeech
- [ ] **Head evaluation script**: Evaluate each head on TIMIT, select best 15
- [ ] **Training loop update**: Add attention loss to training

### Medium Priority

- [ ] **Audio augmentation**: Implement WavLM-style augmentation
- [ ] **Tokenizer adjustment**: Consider splitting tokens on word boundaries
- [ ] **Audio shifting**: 50% probability to shift audio during training

### Nice to Have

- [ ] **Attention clipping**: Clip attention outside ±4 seconds of ground truth
- [ ] **Silence handling**: Prepend space token for silence at start
- [ ] **Pure noise samples**: 1% samples with only noise

---

## References

1. **Whisper timing.py**: https://github.com/openai/whisper/blob/main/whisper/timing.py
2. **Whisper model.py**: https://github.com/openai/whisper/blob/main/whisper/model.py
3. **CrisperWhisper**: https://github.com/nyrahealth/CrisperWhisper
4. **CrisperWhisper Paper**: https://arxiv.org/abs/2408.16589
5. **CrisperWhisper Transformers Fork**: https://github.com/nyrahealth/transformers/tree/crisper_whisper
