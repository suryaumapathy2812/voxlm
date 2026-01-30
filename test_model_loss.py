#!/usr/bin/env python3
"""
Quick test to verify model loss computation is correct.
This bypasses the training script to isolate the issue.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import QwenSTT

print("=" * 60)
print("Testing Qwen-STT Model Forward Pass")
print("=" * 60)

# Load model
print("\n1. Loading model...")
model = QwenSTT.from_pretrained("qwen-stt-0.5b")
model.train()
model = model.cuda()
print("   ✓ Model loaded")

# Create dummy batch
print("\n2. Creating dummy batch...")
batch_size = 2
audio_samples = 16000 * 10  # 10 seconds
audio = torch.randn(batch_size, audio_samples) * 0.1  # Random audio

# Create random labels (50 tokens each)
vocab_size = model.tokenizer.vocab_size
max_seq_len = 50
labels = torch.randint(1, vocab_size, (batch_size, max_seq_len))

# Set some labels to pad_token_id
labels[:, 30:] = model.tokenizer.pad_token_id

print(f"   ✓ Created batch: {batch_size} x {audio_samples} audio, {max_seq_len} tokens")
print(f"   ✓ Vocab size: {vocab_size}")
print(f"   ✓ Pad token ID: {model.tokenizer.pad_token_id}")

# Move to CUDA
audio = audio.cuda()
labels = labels.cuda()

# Forward pass
print("\n3. Running forward pass...")
outputs = model(audio, labels=labels)

# Check outputs
print("\n4. Analyzing outputs...")
print(f"   Loss value:     {outputs.loss.item():.6f}")
print(f"   Logits shape:   {outputs.text_logits.shape}")
print(f"   Labels shape:   {labels.shape}")

# Get predictions
predicted = outputs.text_logits.argmax(dim=-1)
print(f"\n5. First sample (first 10 tokens):")
print(f"   Labels:   {labels[0, :10].tolist()}")
print(f"   Predicted: {predicted[0, :10].tolist()}")

# Decode
text_labels = model.tokenizer.decode(labels[0, :10])
text_pred = model.tokenizer.decode(predicted[0, :10])
print(f"\n   Decoded labels: '{text_labels}'")
print(f"   Decoded pred:  '{text_pred}'")

# Analyze
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

loss_val = outputs.loss.item()

if loss_val < 2.0:
    print(f"❌ LOSS TOO LOW: {loss_val:.6f}")
    print(f"   Expected: 8.0 - 10.0 (random predictions)")
    print(f"   Possible issues:")
    print(f"     - Model is predicting pad tokens only")
    print(f"     - Model is predicting very few tokens")
    print(f"     - Labels are incorrect")
    print(f"     - Bug in loss computation")
elif loss_val < 5.0:
    print(f"⚠️  LOSS LOW: {loss_val:.6f}")
    print(f"   Expected: 8.0 - 10.0 for random")
    print(f"   Model might be partially trained or there's an issue")
elif loss_val > 15.0:
    print(f"⚠️  LOSS HIGH: {loss_val:.6f}")
    print(f"   Expected: 8.0 - 10.0 for random")
    print(f"   Possible issues:")
    print(f"     - Learning rate too high")
    print(f"     - Gradient explosion")
    print(f"     - Data preprocessing issue")
else:
    print(f"✅ LOSS NORMAL: {loss_val:.6f}")
    print(f"   Expected: 8.0 - 10.0 for random predictions")
    print(f"   Model is working correctly!")

# Check if predictions match labels
match_rate = (predicted[:, :10] == labels[:, :10]).float().mean()
print(f"\n   Match rate (first 10 tokens): {match_rate * 100:.1f}%")

if match_rate < 0.2:
    print(f"   ⚠️  Match rate is too low for random data")
    print(f"      Expected: ~10% (1/vocab_size)")
elif match_rate > 0.5:
    print(f"   ⚠️  Match rate is too high for random data")
    print(f"      Model might be overfitting or labels are constant")

print("\n" + "=" * 60)
