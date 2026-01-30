# VoxLM: Lessons Learned & Decision Log

> A living document capturing mistakes, learnings, and architectural decisions.
> Updated: January 2026

---

## Table of Contents

1. [Critical Bugs & Root Causes](#critical-bugs--root-causes)
2. [Debugging Mistakes](#debugging-mistakes)
3. [Architectural Decisions](#architectural-decisions)
4. [Best Practices](#best-practices)
5. [Code Quality Checklist](#code-quality-checklist)
6. [Future Improvements](#future-improvements)

---

## Critical Bugs & Root Causes

### Bug #1: Model Never Generates EOS (Hallucination)

**Symptom**: Model transcribes correctly but continues generating text beyond the audio content ("hallucination").

**Investigation Path** (what we tried, in order):
1. Per-token confidence stopping → Failed (stopped on rare words like "quilter")
2. Consecutive low-confidence stopping → Failed (hallucination had medium confidence)
3. Confidence-drop detection → Partially worked but unreliable
4. Tighter max_length → Worked as band-aid but not a real fix

**Root Cause Discovery**:
```python
# Qwen2 tokenizer defaults:
eos_token_id = 151643  # <|endoftext|>
pad_token_id = 151643  # <|endoftext|>  ← SAME TOKEN!

# Our loss computation:
loss = F.cross_entropy(..., ignore_index=self.tokenizer.pad_token_id)

# Result: EOS is ignored in loss → Model never learns to stop!
```

**The Fix**:
```python
# Use a different pad token
if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
    fim_pad_id = self.tokenizer.convert_tokens_to_ids("<|fim_pad|>")
    if fim_pad_id is not None:
        self.tokenizer.pad_token = "<|fim_pad|>"
        self.tokenizer.pad_token_id = fim_pad_id
    else:
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
```

**Key Lesson**: 
> When a model doesn't generate EOS, it's a **TRAINING problem**, not an inference problem. Always check:
> 1. Is EOS token in training labels?
> 2. Is EOS included in loss computation (not ignored)?
> 3. Is pad_token different from eos_token?

**Files Changed**: `src/model.py`, `scripts/train.py`, `scripts/train_with_timestamps.py`

---

### Bug #2: Confidence Stopping Breaks on Rare Words

**Symptom**: Model stops mid-word on rare words like "quilter" → outputs "mister qu" instead of "mister quilter"

**Root Cause**: Per-token confidence stopping with fixed threshold (0.6). The subword token "ilter" had confidence 0.48 (below threshold) because "quilter" is rare, but it was correct.

**Key Insight**:
```
Token       Confidence    Correct?
--------    ----------    --------
"mist"      0.98          Yes
"er"        0.99          Yes
"qu"        0.91          Yes
"ilter"     0.48  ← LOW   Yes (rare but correct!)
"is"        0.99          Yes
```

**Lesson**: 
> Confidence ≠ Correctness. A model can be:
> - Confidently wrong (hallucination)
> - Uncertainly correct (rare words)
> 
> Per-token confidence stopping is fundamentally flawed for STT.

**Better Approach**: Post-generation validation (Whisper-style) with:
- compression_ratio (detects repetition)
- words_per_second (sanity check)
- avg_logprob (overall confidence)

---

## Debugging Mistakes

### Mistake #1: Trying to Fix Training Bugs at Inference Time

**What we did**: Spent hours implementing various stopping criteria (confidence-based, consecutive low-confidence, confidence-drop detection) to prevent hallucination.

**Why it was wrong**: The model wasn't trained to stop. No amount of inference tricks can make a model do something it wasn't trained to do.

**What we should have done**: 
1. Check if EOS is in training data
2. Check if EOS is in loss computation
3. Verify tokenizer setup (pad ≠ eos)

**Time wasted**: ~4 hours

---

### Mistake #2: Multiple Commits to Main During Debugging

**What we did**: Made several experimental commits to main branch while testing different stopping strategies.

**Why it was wrong**: Polluted git history with incomplete/experimental changes.

**What we should have done**: 
1. Create feature branch: `git checkout -b fix/hallucination`
2. Experiment on branch
3. Once fixed, squash merge to main

**Commits that should have been on a branch**:
- "refactor: replace per-token confidence stopping..."
- "fix: tighten max_length and words_per_second thresholds..."
- "fix: further tighten thresholds..."

---

### Mistake #3: Not Validating Tokenizer Setup

**What we did**: Assumed tokenizer defaults were sensible.

**Why it was wrong**: Many LLM tokenizers use eos_token as pad_token by default. This is fine for text generation but breaks STT training.

**What we should have done**: Add validation in model initialization:
```python
def __init__(self, config):
    ...
    # VALIDATE tokenizer setup
    assert self.tokenizer.pad_token_id != self.tokenizer.eos_token_id, \
        "CRITICAL: pad_token must differ from eos_token for EOS learning!"
```

---

## Architectural Decisions

### Decision #1: Encoder-LLM Modular Design

**Context**: Need to support multiple audio encoders and LLMs.

**Options Considered**:
1. Monolithic model (single encoder-decoder)
2. Modular design (pluggable encoder + LLM)
3. Adapter-based (frozen models + small adapters)

**Decision**: Option 2 + 3 (Modular + Adapters)

**Rationale**:
- Encoder-agnostic: Can swap Whisper/IndicWhisper/MMS
- LLM-agnostic: Can swap Qwen/Llama/Phi
- LoRA adapters: Train efficiently without full fine-tuning
- Projection layer: Bridges different embedding dimensions

**Trade-offs**:
- (+) Flexibility, easy experimentation
- (+) Efficient training with LoRA
- (-) More complex than monolithic
- (-) Potential embedding space mismatch

---

### Decision #2: Two-Phase Training

**Context**: Need to learn both transcription and timestamp alignment.

**Options Considered**:
1. Joint training (transcription + timestamps together)
2. Sequential training (Phase 1: transcription, Phase 2: timestamps)
3. Multi-task learning with task weights

**Decision**: Option 2 (Sequential)

**Rationale**:
- Phase 1: Model learns audio→text mapping
- Phase 2: Model learns alignment (with frozen transcription ability)
- Simpler to debug and tune
- Can use Phase 1 model standalone if timestamps not needed

**Trade-offs**:
- (+) Easier to debug each phase
- (+) Can stop after Phase 1 for basic STT
- (-) May not be globally optimal
- (-) Longer total training time

---

### Decision #3: Whisper-Style Post-Validation (Defense in Depth)

**Context**: Need to prevent hallucination/garbage output at inference time.

**Options Considered**:
1. Per-token confidence stopping
2. Post-generation validation + retry
3. Beam search with length penalty
4. No stopping (rely on EOS)

**Decision**: Option 4 (primary) + Option 2 (backup)

**Rationale**:
- Primary: Model should learn to generate EOS (fixed via tokenizer)
- Backup: TranscriptionValidator catches edge cases
- Metrics: compression_ratio, words_per_second, avg_logprob
- Temperature fallback: Retry with higher temperature if quality poor

**Implementation**: `TranscriptionValidator` class in `src/model.py`

---

## Best Practices

### For LLM-Based STT

1. **Always verify tokenizer setup**:
   ```python
   assert tokenizer.pad_token_id != tokenizer.eos_token_id
   ```

2. **Include EOS in training labels**:
   ```python
   texts = [t + tokenizer.eos_token for t in texts]
   ```

3. **Don't ignore EOS in loss**:
   ```python
   # BAD (if pad==eos): 
   loss = F.cross_entropy(..., ignore_index=tokenizer.pad_token_id)
   
   # GOOD: Use different pad token
   tokenizer.pad_token = "<|pad|>"
   ```

4. **Test EOS generation**:
   ```python
   # Model should stop before max_length
   output = model.generate(audio, max_new_tokens=100)
   assert tokenizer.eos_token_id in output
   ```

### For Debugging

1. **Think training before inference**: If model doesn't do X, first ask "was it trained to do X?"

2. **Check the data pipeline**: 
   - Print sample labels
   - Verify EOS is present
   - Check tokenization

3. **Use feature branches**: Don't experiment on main

4. **Add assertions**: Fail fast on invalid states

### For Code Quality

1. **Remove debug prints before committing**
2. **Add docstrings explaining "why", not just "what"**
3. **Write tests for critical paths (tokenizer, loss, etc.)**

---

## Code Quality Checklist

Before committing, verify:

- [ ] No debug print statements
- [ ] Tokenizer: pad_token ≠ eos_token
- [ ] Training labels include EOS
- [ ] Loss doesn't ignore EOS
- [ ] Tests pass
- [ ] Docstrings updated
- [ ] No hardcoded paths

Before training, verify:

- [ ] Print tokenizer setup (eos/pad tokens)
- [ ] Print sample batch (verify EOS in labels)
- [ ] Verify loss is decreasing
- [ ] Monitor for NaN/Inf

Before inference, verify:

- [ ] Model generates EOS (test without max_length restriction)
- [ ] Output quality metrics (WER, compression_ratio)
- [ ] No repetition loops

---

## Future Improvements

### High Priority

| Item | Why | Status |
|------|-----|--------|
| Unit tests for tokenizer | Prevent regression | TODO |
| WER evaluation script | Objective quality measurement | TODO |
| Clean up experimental code | Production readiness | TODO |

### Medium Priority

| Item | Why | Status |
|------|-----|--------|
| Multi-dataset training | Robustness | TODO |
| Streaming inference | Real-time use cases | TODO |
| Quantization (INT8/INT4) | Faster inference | TODO |

### Low Priority

| Item | Why | Status |
|------|-----|--------|
| Multi-language (IndicWhisper) | Indian language support | TODO |
| Speaker diarization | "Who said what" | TODO |
| Web demo | Easy testing | TODO |

---

## References

- [Whisper Paper](https://arxiv.org/abs/2212.04356) - OpenAI's approach to STT
- [HuggingFace Generation Strategies](https://huggingface.co/blog/how-to-generate)
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-31 | Initial document | VoxLM Team |
| 2026-01-31 | Added EOS bug analysis | VoxLM Team |

