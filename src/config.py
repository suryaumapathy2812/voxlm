"""
Configuration classes for VoxLM models.

VoxLM: Modular Speech-to-Text with LLM intelligence.
Supports any audio encoder (Whisper, IndicWhisper, MMS) + any LLM (Qwen, Llama, Phi).

Supports loading from:
- Pre-defined model names: get_config("voxlm-2b")
- YAML config files: load_config_from_yaml("configs/voxlm-2b.yaml")
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List, Tuple, Dict, Any
from pathlib import Path
import copy


@dataclass
class VoxLMConfig:
    """Configuration for VoxLM model.

    VoxLM is a modular architecture that combines:
    - Audio Encoder: Whisper, IndicWhisper, MMS, etc.
    - LLM Backbone: Qwen, Llama, Phi, Gemma, etc.
    - Alignment Module: For word-level timestamps
    - Confidence Extraction: From LLM token probabilities
    """

    # Model variant name
    name: str = "voxlm-2b"

    # Architecture version: "v1" (old heads) or "v2" (alignment module)
    architecture_version: str = "v2"

    # ==========================================================================
    # Audio Encoder Configuration
    # Supports: Whisper (any size), IndicWhisper, MMS, Wav2Vec2, HuBERT
    # ==========================================================================
    audio_encoder: str = "openai/whisper-small"  # HuggingFace model ID
    audio_encoder_dim: int = 768  # Output dimension of encoder
    freeze_audio_encoder: bool = True

    # Legacy aliases (for backward compatibility)
    @property
    def whisper_model(self) -> str:
        return self.audio_encoder

    @property
    def whisper_dim(self) -> int:
        return self.audio_encoder_dim

    # ==========================================================================
    # LLM Backbone Configuration
    # Supports: Qwen2/2.5, Llama 3/3.1, Phi-3/4, Gemma 2, Mistral, etc.
    # ==========================================================================
    llm_model: str = "Qwen/Qwen2-1.5B"  # HuggingFace model ID
    llm_dim: int = 1536  # Hidden dimension of LLM
    freeze_llm: bool = True

    # LoRA configuration (for efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # Auto-detect if None

    # ==========================================================================
    # Projection Layer Configuration
    # Bridges audio encoder output to LLM embedding space
    # ==========================================================================
    projection_dropout: float = 0.1
    downsample_factor: int = 4  # 50Hz -> 12.5Hz

    # ==========================================================================
    # Alignment Module Configuration (v2)
    # Cross-attention for word-level timestamps
    # ==========================================================================
    alignment_num_layers: int = 2
    alignment_num_heads: int = 8
    alignment_dropout: float = 0.1
    alignment_loss_weight: float = 1.0
    alignment_interp_frames: int = 4  # Linear interpolation at boundaries
    monotonicity_loss_weight: float = 0.1
    alignment_heads: Optional[List[Tuple[int, int]]] = None  # Head selection

    # ==========================================================================
    # Timestamp Extraction Configuration
    # ==========================================================================
    median_filter_width: int = 7
    max_audio_frames: int = 750  # 60s at 12.5Hz

    # ==========================================================================
    # Confidence Extraction Configuration
    # ==========================================================================
    confidence_aggregation: str = "min"  # "min", "mean", or "product"
    confidence_temperature: float = 1.0

    # ==========================================================================
    # Audio Settings
    # ==========================================================================
    sample_rate: int = 16000
    frame_rate: int = 50  # Whisper outputs 50Hz (20ms frames)
    effective_frame_rate: float = 12.5  # After 4x downsampling (80ms frames)

    # ==========================================================================
    # Training Settings
    # ==========================================================================
    max_audio_length: int = 30  # seconds
    max_text_length: int = 448  # tokens

    # ==========================================================================
    # Special Tokens
    # ==========================================================================
    audio_start_token: str = "<|audio|>"
    audio_end_token: str = "<|/audio|>"
    transcribe_token: str = "<|transcribe|>"

    # ==========================================================================
    # Legacy v1 Configuration (deprecated)
    # ==========================================================================
    timestamp_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.5

    @property
    def frame_duration_ms(self) -> float:
        """Duration of each frame in milliseconds after downsampling."""
        return 1000.0 / self.effective_frame_rate  # 80ms for 12.5Hz


# =============================================================================
# Pre-defined Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    # =========================================================================
    # POC Models (Small, for testing)
    # =========================================================================
    "voxlm-0.5b": VoxLMConfig(
        name="voxlm-0.5b",
        architecture_version="v2",
        audio_encoder="openai/whisper-tiny",
        audio_encoder_dim=384,
        llm_model="Qwen/Qwen2-0.5B",
        llm_dim=896,
        downsample_factor=4,
        max_audio_frames=375,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
    ),
    "voxlm-2b": VoxLMConfig(
        name="voxlm-2b",
        architecture_version="v2",
        audio_encoder="openai/whisper-small",
        audio_encoder_dim=768,
        llm_model="Qwen/Qwen2-1.5B",
        llm_dim=1536,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
    ),
    # =========================================================================
    # Multilingual POC (for testing multilingual support)
    # =========================================================================
    "voxlm-4b-multilingual": VoxLMConfig(
        name="voxlm-4b-multilingual",
        architecture_version="v2",
        audio_encoder="openai/whisper-large-v3-turbo",
        audio_encoder_dim=1280,
        llm_model="Qwen/Qwen2.5-3B",
        llm_dim=2048,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
    ),
    # =========================================================================
    # Production Models - Global (99 languages)
    # =========================================================================
    "voxlm-9b-global": VoxLMConfig(
        name="voxlm-9b-global",
        architecture_version="v2",
        audio_encoder="openai/whisper-large-v3",
        audio_encoder_dim=1280,
        llm_model="Qwen/Qwen2.5-7B",
        llm_dim=3584,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
        lora_r=32,  # Larger LoRA for production
        lora_alpha=64,
    ),
    # =========================================================================
    # Production Models - India (12+ Indian languages)
    # =========================================================================
    "voxlm-9b-india": VoxLMConfig(
        name="voxlm-9b-india",
        architecture_version="v2",
        audio_encoder="ai4bharat/indicwhisper-large-v2",  # IndicWhisper
        audio_encoder_dim=1280,
        llm_model="Qwen/Qwen2.5-7B",
        llm_dim=3584,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
        lora_r=32,
        lora_alpha=64,
    ),
    # =========================================================================
    # Alternative LLM Variants
    # =========================================================================
    "voxlm-llama-8b": VoxLMConfig(
        name="voxlm-llama-8b",
        architecture_version="v2",
        audio_encoder="openai/whisper-large-v3",
        audio_encoder_dim=1280,
        llm_model="meta-llama/Llama-3.1-8B-Instruct",
        llm_dim=4096,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
        lora_r=32,
        lora_alpha=64,
    ),
    "voxlm-phi-4": VoxLMConfig(
        name="voxlm-phi-4",
        architecture_version="v2",
        audio_encoder="openai/whisper-large-v3",
        audio_encoder_dim=1280,
        llm_model="microsoft/phi-4",
        llm_dim=5120,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
        lora_r=32,
        lora_alpha=64,
    ),
    "voxlm-gemma-9b": VoxLMConfig(
        name="voxlm-gemma-9b",
        architecture_version="v2",
        audio_encoder="openai/whisper-large-v3",
        audio_encoder_dim=1280,
        llm_model="google/gemma-2-9b-it",
        llm_dim=3584,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
        lora_r=32,
        lora_alpha=64,
    ),
    # =========================================================================
    # Edge/Mobile Models (Small, fast inference)
    # =========================================================================
    "voxlm-edge": VoxLMConfig(
        name="voxlm-edge",
        architecture_version="v2",
        audio_encoder="openai/whisper-tiny",
        audio_encoder_dim=384,
        llm_model="Qwen/Qwen2.5-0.5B",
        llm_dim=896,
        downsample_factor=4,
        max_audio_frames=375,
        effective_frame_rate=12.5,
        alignment_num_heads=4,
        lora_r=8,
    ),
    # =========================================================================
    # Legacy Configurations (backward compatibility)
    # =========================================================================
    # Old qwen-stt names map to new voxlm names
    "qwen-stt-0.5b": VoxLMConfig(
        name="voxlm-0.5b",
        architecture_version="v2",
        audio_encoder="openai/whisper-tiny",
        audio_encoder_dim=384,
        llm_model="Qwen/Qwen2-0.5B",
        llm_dim=896,
        downsample_factor=4,
        max_audio_frames=375,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
    ),
    "qwen-stt-2b": VoxLMConfig(
        name="voxlm-2b",
        architecture_version="v2",
        audio_encoder="openai/whisper-small",
        audio_encoder_dim=768,
        llm_model="Qwen/Qwen2-1.5B",
        llm_dim=1536,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
    ),
    "qwen-stt-7b": VoxLMConfig(
        name="voxlm-7b",
        architecture_version="v2",
        audio_encoder="openai/whisper-medium",
        audio_encoder_dim=1024,
        llm_model="Qwen/Qwen2-7B",
        llm_dim=3584,
        downsample_factor=4,
        max_audio_frames=750,
        effective_frame_rate=12.5,
        alignment_num_heads=8,
    ),
}


def get_config(model_name: str) -> VoxLMConfig:
    """Get configuration for a model variant.

    Args:
        model_name: Name of the model configuration.
                   Examples: "voxlm-2b", "voxlm-9b-global", "voxlm-9b-india"

    Returns:
        VoxLMConfig instance

    Raises:
        ValueError: If model_name is not found
    """
    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown model: {model_name}.\n"
            f"Available models:\n"
            f"  POC: voxlm-0.5b, voxlm-2b\n"
            f"  Multilingual: voxlm-4b-multilingual\n"
            f"  Production: voxlm-9b-global, voxlm-9b-india\n"
            f"  Alternatives: voxlm-llama-8b, voxlm-phi-4, voxlm-gemma-9b\n"
            f"  Edge: voxlm-edge\n"
            f"  Legacy: qwen-stt-0.5b, qwen-stt-2b, qwen-stt-7b"
        )
    return MODEL_CONFIGS[model_name]


# Backward compatibility alias
QwenSTTConfig = VoxLMConfig


# =============================================================================
# YAML Configuration Loading
# =============================================================================


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config file with inheritance support.

    Supports _base_ key for inheriting from another config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Merged configuration dictionary
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config loading. "
            "Install with: pip install pyyaml"
        )

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if "_base_" in config:
        base_path = config_file.parent / config["_base_"]
        base_config = load_yaml_config(str(base_path))
        del config["_base_"]
        config = _deep_merge(base_config, config)

    return config


def load_config_from_yaml(config_path: str) -> VoxLMConfig:
    """
    Load VoxLMConfig from a YAML file.

    The YAML file should have a 'model' section with configuration values.
    Supports inheritance via _base_ key.

    Args:
        config_path: Path to YAML config file

    Returns:
        VoxLMConfig instance

    Example:
        config = load_config_from_yaml("configs/voxlm-2b.yaml")
        model = VoxLM(config)
    """
    yaml_config = load_yaml_config(config_path)

    # Extract model configuration
    model_config = yaml_config.get("model", {})

    # Map YAML keys to VoxLMConfig fields
    config_kwargs = {}

    # Direct mappings
    direct_fields = [
        "name",
        "architecture_version",
        "audio_encoder",
        "audio_encoder_dim",
        "freeze_audio_encoder",
        "llm_model",
        "llm_dim",
        "freeze_llm",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "projection_dropout",
        "downsample_factor",
        "alignment_num_layers",
        "alignment_num_heads",
        "alignment_dropout",
        "alignment_loss_weight",
        "alignment_interp_frames",
        "monotonicity_loss_weight",
        "alignment_heads",
        "median_filter_width",
        "max_audio_frames",
        "confidence_aggregation",
        "confidence_temperature",
        "sample_rate",
        "frame_rate",
        "effective_frame_rate",
        "max_audio_length",
        "max_text_length",
        "audio_start_token",
        "audio_end_token",
        "transcribe_token",
        "timestamp_loss_weight",
        "confidence_loss_weight",
    ]

    for field in direct_fields:
        if field in model_config:
            config_kwargs[field] = model_config[field]

    return VoxLMConfig(**config_kwargs)


@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML."""

    # Phase 1: Basic transcription
    phase1_epochs: int = 10
    phase1_batch_size: int = 8
    phase1_lr: float = 1e-4
    phase1_warmup_steps: int = 1000
    phase1_grad_accum: int = 1

    # Phase 2: Timestamp alignment
    phase2_epochs: int = 5
    phase2_batch_size: int = 16
    phase2_lr: float = 5e-5
    phase2_alignment_weight: float = 1.0

    # Common
    max_audio_length: int = 30
    max_text_length: int = 448
    use_amp: bool = True
    compile_model: bool = False
    num_workers: int = 8


@dataclass
class DataConfig:
    """Data configuration loaded from YAML."""

    data_dir: str = "./data"
    train_split: str = "train-clean-100"
    val_split: str = "dev-clean"
    timestamps_dir: str = "./data/timestamps"
    whisper_model: str = "small"
    max_wer: float = 0.1


@dataclass
class OutputConfig:
    """Output configuration loaded from YAML."""

    checkpoint_dir: str = "./checkpoints"
    final_model_dir: str = "./output/models"
    use_wandb: bool = False
    wandb_project: str = "voxlm"


@dataclass
class FullConfig:
    """Complete configuration combining model, training, data, and output."""

    model: VoxLMConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig
    device: str = "auto"


def load_full_config(config_path: str) -> FullConfig:
    """
    Load complete configuration from YAML file.

    Returns a FullConfig with model, training, data, and output settings.

    Args:
        config_path: Path to YAML config file

    Returns:
        FullConfig instance with all settings

    Example:
        config = load_full_config("configs/voxlm-2b.yaml")
        print(config.model.name)  # "voxlm-2b"
        print(config.training.phase1_epochs)  # 10
        print(config.data.train_split)  # "train-clean-100"
    """
    yaml_config = load_yaml_config(config_path)

    # Load model config
    model_config = load_config_from_yaml(config_path)

    # Load training config
    training_yaml = yaml_config.get("training", {})
    phase1 = training_yaml.get("phase1", {})
    phase2 = training_yaml.get("phase2", {})

    training_config = TrainingConfig(
        phase1_epochs=phase1.get("epochs", 10),
        phase1_batch_size=phase1.get("batch_size", 8),
        phase1_lr=phase1.get("learning_rate", 1e-4),
        phase1_warmup_steps=phase1.get("warmup_steps", 1000),
        phase1_grad_accum=phase1.get("gradient_accumulation", 1),
        phase2_epochs=phase2.get("epochs", 5),
        phase2_batch_size=phase2.get("batch_size", 16),
        phase2_lr=phase2.get("learning_rate", 5e-5),
        phase2_alignment_weight=phase2.get("alignment_weight", 1.0),
        max_audio_length=training_yaml.get("max_audio_length", 30),
        max_text_length=training_yaml.get("max_text_length", 448),
        use_amp=training_yaml.get("use_amp", True),
        compile_model=training_yaml.get("compile_model", False),
        num_workers=training_yaml.get("num_workers", 8),
    )

    # Load data config
    data_yaml = yaml_config.get("data", {})
    data_config = DataConfig(
        data_dir=data_yaml.get("data_dir", "./data"),
        train_split=data_yaml.get("train_split", "train-clean-100"),
        val_split=data_yaml.get("val_split", "dev-clean"),
        timestamps_dir=data_yaml.get("timestamps_dir", "./data/timestamps"),
        whisper_model=data_yaml.get("whisper_model", "small"),
        max_wer=data_yaml.get("max_wer", 0.1),
    )

    # Load output config
    output_yaml = yaml_config.get("output", {})
    output_config = OutputConfig(
        checkpoint_dir=output_yaml.get("checkpoint_dir", "./checkpoints"),
        final_model_dir=output_yaml.get("final_model_dir", "./output/models"),
        use_wandb=output_yaml.get("use_wandb", False),
        wandb_project=output_yaml.get("wandb_project", "voxlm"),
    )

    # Load device config
    device_yaml = yaml_config.get("device", {})
    device = device_yaml.get("type", "auto")

    return FullConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        output=output_config,
        device=device,
    )
