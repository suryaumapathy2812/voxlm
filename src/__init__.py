"""
VoxLM: Modular Speech-to-Text with LLM Intelligence

A flexible architecture combining any audio encoder + any LLM:
- AudioEncoder: Whisper, IndicWhisper, MMS, Wav2Vec2, etc.
- VoxLM: Main model combining audio encoder + LLM
- AlignmentModule: Cross-attention for audio-text alignment
  CRITICAL: Uses PURE audio embeddings, not LLM hidden states!
- TimestampExtractor: DTW-based timestamp extraction
- ConfidenceExtractor: Token probability confidence (no extra params)
- AttentionAlignmentLoss: CrisperWhisper-style cosine similarity loss

Use Cases:
- Global: Whisper-large-v3 + Qwen2.5-7B (99 languages)
- India: IndicWhisper + Qwen2.5-7B (12+ Indian languages)
- Edge: Whisper-tiny + Qwen2.5-0.5B (fast inference)

Architecture v1 (legacy - DO NOT USE):
- TimestampHead: Frame classification (fundamentally flawed)
- ConfidenceHead: Separate confidence head (no training signal)
"""

from .config import (
    VoxLMConfig,
    MODEL_CONFIGS,
    get_config,
    # YAML config loading
    load_config_from_yaml,
    load_full_config,
    load_yaml_config,
    TrainingConfig,
    DataConfig,
    OutputConfig,
    FullConfig,
)
from .audio_encoder import AudioEncoder, AudioProjection
from .model import VoxLM, VoxLMOutput

# v1 components (legacy - deprecated)
from .heads import TimestampHead, ConfidenceHead, AttentionBasedTimestamps

# v2 components (recommended)
from .alignment import (
    AlignmentModule,
    CrossAttentionBlock,
    TimestampExtractor,
    ConfidenceExtractor,
    AttentionAlignmentLoss,
    alignment_loss,
    alignment_loss_from_frame_indices,
    monotonicity_loss,
    merge_timestamps_and_confidence,
)

# Backward compatibility aliases
QwenSTTConfig = VoxLMConfig
QwenSTT = VoxLM
QwenSTTOutput = VoxLMOutput

__version__ = "0.4.0"  # YAML config support
__all__ = [
    # Config
    "VoxLMConfig",
    "MODEL_CONFIGS",
    "get_config",
    # YAML config loading
    "load_config_from_yaml",
    "load_full_config",
    "load_yaml_config",
    "TrainingConfig",
    "DataConfig",
    "OutputConfig",
    "FullConfig",
    # Core
    "AudioEncoder",
    "AudioProjection",
    "VoxLM",
    "VoxLMOutput",
    # v1 (legacy - deprecated)
    "TimestampHead",
    "ConfidenceHead",
    "AttentionBasedTimestamps",
    # v2 (recommended)
    "AlignmentModule",
    "CrossAttentionBlock",
    "TimestampExtractor",
    "ConfidenceExtractor",
    "AttentionAlignmentLoss",
    "alignment_loss",
    "alignment_loss_from_frame_indices",
    "monotonicity_loss",
    "merge_timestamps_and_confidence",
    # Backward compatibility
    "QwenSTTConfig",
    "QwenSTT",
    "QwenSTTOutput",
]
