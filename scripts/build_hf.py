#!/usr/bin/env python3
"""
Build HuggingFace-compatible model files from VoxLM source.

This script:
1. Merges source files into a self-contained modeling_voxlm.py
2. Creates configuration_voxlm.py with PretrainedConfig
3. Converts model.pt to model.safetensors
4. Generates config.json with auto_map for AutoModel
5. Copies README.md
6. Optionally uploads to HuggingFace

Usage:
    # Build only
    python scripts/build_hf.py --checkpoint models/voxlm-2b/model.pt --output hf/

    # Build and upload
    python scripts/build_hf.py --checkpoint models/voxlm-2b/model.pt --output hf/ --upload suryaumapathy2812/voxlm-2b
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def merge_source_files(output_path: Path) -> None:
    """
    Merge VoxLM source files into a single modeling_voxlm.py.

    This creates a self-contained file that can be loaded by HuggingFace
    with trust_remote_code=True.
    """
    src_dir = Path(__file__).parent.parent / "src"

    # Read source files
    with open(src_dir / "alignment.py") as f:
        alignment_code = f.read()

    with open(src_dir / "audio_encoder.py") as f:
        audio_encoder_code = f.read()

    with open(src_dir / "heads.py") as f:
        heads_code = f.read()

    with open(src_dir / "model.py") as f:
        model_code = f.read()

    # Process alignment.py - remove imports we'll consolidate
    alignment_lines = []
    for line in alignment_code.split("\n"):
        # Keep the line unless it's an import we're consolidating
        if line.startswith("from .") or line.startswith("import ."):
            continue
        alignment_lines.append(line)
    alignment_processed = "\n".join(alignment_lines)

    # Process audio_encoder.py
    audio_encoder_lines = []
    for line in audio_encoder_code.split("\n"):
        if line.startswith("from .") or line.startswith("import ."):
            continue
        audio_encoder_lines.append(line)
    audio_encoder_processed = "\n".join(audio_encoder_lines)

    # Process heads.py
    heads_lines = []
    for line in heads_code.split("\n"):
        if line.startswith("from .") or line.startswith("import ."):
            continue
        heads_lines.append(line)
    heads_processed = "\n".join(heads_lines)

    # Process model.py - remove relative imports
    model_lines = []
    skip_imports = [
        "from .config import",
        "from .audio_encoder import",
        "from .heads import",
        "from .alignment import",
    ]
    for line in model_code.split("\n"):
        skip = False
        for skip_import in skip_imports:
            if line.strip().startswith(skip_import):
                skip = True
                break
        if not skip:
            model_lines.append(line)
    model_processed = "\n".join(model_lines)

    # Build the merged file
    merged_content = '''"""
VoxLM: Modular Speech-to-Text with LLM Intelligence.

This is an auto-generated file for HuggingFace compatibility.
Do not edit directly - regenerate with: python scripts/build_hf.py

Original source files:
- src/alignment.py
- src/audio_encoder.py  
- src/heads.py
- src/model.py
"""

# =============================================================================
# Imports
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import zlib
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    WhisperModel,
    WhisperFeatureExtractor,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

# Try to import numba for JIT compilation (optional but recommended)
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Import configuration from the same directory
from .configuration_voxlm import VoxLMConfig


# =============================================================================
# Audio Encoder
# =============================================================================

'''

    # Add audio encoder (remove its imports and docstring header)
    audio_lines = audio_encoder_processed.split("\n")
    in_header = True
    for i, line in enumerate(audio_lines):
        if in_header and (
            line.startswith("import ")
            or line.startswith("from ")
            or line.strip().startswith('"""')
            or line.strip() == ""
            or line.startswith("#")
        ):
            continue
        if line.startswith("class ") or line.startswith("def "):
            in_header = False
        if not in_header:
            merged_content += line + "\n"

    merged_content += """

# =============================================================================
# Output Heads (v1, deprecated)
# =============================================================================

"""

    # Add heads (remove its imports and docstring header)
    heads_lines_list = heads_processed.split("\n")
    in_header = True
    for i, line in enumerate(heads_lines_list):
        if in_header and (
            line.startswith("import ")
            or line.startswith("from ")
            or line.strip().startswith('"""')
            or line.strip() == ""
            or line.startswith("#")
        ):
            continue
        if line.startswith("class ") or line.startswith("def "):
            in_header = False
        if not in_header:
            merged_content += line + "\n"

    merged_content += """

# =============================================================================
# Alignment Module (v2)
# =============================================================================

"""

    # Add alignment (remove its imports and docstring header)
    alignment_lines_list = alignment_processed.split("\n")
    in_header = True
    for i, line in enumerate(alignment_lines_list):
        if in_header and (
            line.startswith("import ")
            or line.startswith("from ")
            or line.strip().startswith('"""')
            or line.strip() == ""
            or line.startswith("#")
            or "HAS_NUMBA" in line
        ):
            continue
        if (
            line.startswith("class ")
            or line.startswith("def ")
            or line.startswith("if HAS_NUMBA")
        ):
            in_header = False
        if not in_header:
            merged_content += line + "\n"

    merged_content += """

# =============================================================================
# VoxLM Model
# =============================================================================

"""

    # Add model (remove its imports and docstring header)
    model_lines_list = model_processed.split("\n")
    in_header = True
    for i, line in enumerate(model_lines_list):
        if in_header and (
            line.startswith("import ")
            or line.startswith("from ")
            or line.strip().startswith('"""')
            or line.strip() == ""
            or line.startswith("#")
        ):
            continue
        if line.startswith("class ") or line.startswith("def ") or line.startswith("@"):
            in_header = False
        if not in_header:
            merged_content += line + "\n"

    # Add HuggingFace-compatible wrapper class
    merged_content += '''

# =============================================================================
# HuggingFace-Compatible Wrapper
# =============================================================================


class VoxLMForConditionalGeneration(VoxLM, PreTrainedModel):
    """
    HuggingFace-compatible wrapper for VoxLM.
    
    Enables loading with:
        model = AutoModel.from_pretrained("repo/voxlm-2b", trust_remote_code=True)
    """
    
    config_class = VoxLMConfig
    base_model_prefix = "voxlm"
    supports_gradient_checkpointing = False
    
    def __init__(self, config: VoxLMConfig):
        # Initialize PreTrainedModel first
        PreTrainedModel.__init__(self, config)
        # Then initialize VoxLM
        VoxLM.__init__(self, config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load model from HuggingFace Hub or local path."""
        from transformers import AutoConfig
        from safetensors.torch import load_file
        from pathlib import Path
        import os
        
        # Remove trust_remote_code from kwargs to avoid duplicate argument error
        kwargs.pop('trust_remote_code', None)
        
        # Load config
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **kwargs
        )
        
        # Create model
        model = cls(config)
        
        # Load weights
        model_path = Path(pretrained_model_name_or_path)
        if model_path.is_dir():
            # Local directory
            safetensors_path = model_path / "model.safetensors"
            pt_path = model_path / "model.pt"
            
            if safetensors_path.exists():
                state_dict = load_file(str(safetensors_path))
                model.load_state_dict(state_dict, strict=False)
            elif pt_path.exists():
                checkpoint = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
        else:
            # HuggingFace Hub
            from huggingface_hub import hf_hub_download
            
            try:
                safetensors_path = hf_hub_download(
                    pretrained_model_name_or_path,
                    "model.safetensors"
                )
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pt_path = hf_hub_download(
                    pretrained_model_name_or_path,
                    "model.pt"
                )
                checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
        
        return model
    
    def generate(
        self,
        audio: torch.Tensor,
        instruction: Optional[str] = None,
        max_new_tokens: int = 448,
        **kwargs
    ) -> Dict:
        """
        HuggingFace-compatible generate method.
        
        This wraps the transcribe() method for HF compatibility.
        
        Args:
            audio: Raw audio waveform [samples] or [1, samples] at 16kHz
            instruction: Optional text instruction/context
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to transcribe()
        
        Returns:
            Dict with 'text', 'words' (timestamps and confidence)
        """
        return self.transcribe(
            audio=audio,
            instruction=instruction,
            max_length=max_new_tokens,
            **kwargs
        )


# Alias for AutoModel registration
VoxLMModel = VoxLMForConditionalGeneration
'''

    # Write merged file
    with open(output_path / "modeling_voxlm.py", "w") as f:
        f.write(merged_content)

    print(f"Created: {output_path / 'modeling_voxlm.py'}")


def create_configuration_file(output_path: Path) -> None:
    """
    Create configuration_voxlm.py with HuggingFace PretrainedConfig.
    """
    config_content = '''"""
VoxLM Configuration for HuggingFace.

This is an auto-generated file for HuggingFace compatibility.
Do not edit directly - regenerate with: python scripts/build_hf.py
"""

from transformers import PretrainedConfig
from typing import Optional, List, Tuple


class VoxLMConfig(PretrainedConfig):
    """
    Configuration class for VoxLM model.
    
    VoxLM is a modular Speech-to-Text architecture combining:
    - Audio Encoder: Whisper, IndicWhisper, MMS, etc.
    - LLM Backbone: Qwen, Llama, Phi, Gemma, etc.
    - Alignment Module: For word-level timestamps
    - Confidence Extraction: From LLM token probabilities
    """
    
    model_type = "voxlm"
    
    def __init__(
        self,
        # Model variant name
        name: str = "voxlm-2b",
        
        # Architecture version: "v1" (old heads) or "v2" (alignment module)
        architecture_version: str = "v2",
        
        # Audio Encoder Configuration
        audio_encoder: str = "openai/whisper-small",
        audio_encoder_dim: int = 768,
        freeze_audio_encoder: bool = True,
        
        # LLM Backbone Configuration
        llm_model: str = "Qwen/Qwen2-1.5B",
        llm_dim: int = 1536,
        freeze_llm: bool = True,
        
        # LoRA configuration
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        
        # Projection Layer Configuration
        projection_dropout: float = 0.1,
        downsample_factor: int = 4,
        
        # Alignment Module Configuration (v2)
        alignment_num_layers: int = 2,
        alignment_num_heads: int = 8,
        alignment_dropout: float = 0.1,
        alignment_loss_weight: float = 1.0,
        alignment_interp_frames: int = 4,
        monotonicity_loss_weight: float = 0.1,
        alignment_heads: Optional[List[Tuple[int, int]]] = None,
        
        # Timestamp Extraction Configuration
        median_filter_width: int = 7,
        max_audio_frames: int = 750,
        
        # Confidence Extraction Configuration
        confidence_aggregation: str = "min",
        confidence_temperature: float = 1.0,
        
        # Audio Settings
        sample_rate: int = 16000,
        frame_rate: int = 50,
        effective_frame_rate: float = 12.5,
        
        # Training Settings
        max_audio_length: int = 30,
        max_text_length: int = 448,
        
        # Special Tokens
        audio_start_token: str = "<|audio|>",
        audio_end_token: str = "<|/audio|>",
        transcribe_token: str = "<|transcribe|>",
        
        # Legacy v1 Configuration
        timestamp_loss_weight: float = 1.0,
        confidence_loss_weight: float = 0.5,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.name = name
        self.architecture_version = architecture_version
        
        # Audio encoder
        self.audio_encoder = audio_encoder
        self.audio_encoder_dim = audio_encoder_dim
        self.freeze_audio_encoder = freeze_audio_encoder
        
        # LLM
        self.llm_model = llm_model
        self.llm_dim = llm_dim
        self.freeze_llm = freeze_llm
        
        # LoRA
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        
        # Projection
        self.projection_dropout = projection_dropout
        self.downsample_factor = downsample_factor
        
        # Alignment
        self.alignment_num_layers = alignment_num_layers
        self.alignment_num_heads = alignment_num_heads
        self.alignment_dropout = alignment_dropout
        self.alignment_loss_weight = alignment_loss_weight
        self.alignment_interp_frames = alignment_interp_frames
        self.monotonicity_loss_weight = monotonicity_loss_weight
        self.alignment_heads = alignment_heads
        
        # Timestamps
        self.median_filter_width = median_filter_width
        self.max_audio_frames = max_audio_frames
        
        # Confidence
        self.confidence_aggregation = confidence_aggregation
        self.confidence_temperature = confidence_temperature
        
        # Audio
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.effective_frame_rate = effective_frame_rate
        
        # Training
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
        # Special tokens
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.transcribe_token = transcribe_token
        
        # Legacy
        self.timestamp_loss_weight = timestamp_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
    
    @property
    def whisper_model(self) -> str:
        """Legacy alias for audio_encoder."""
        return self.audio_encoder
    
    @property
    def whisper_dim(self) -> int:
        """Legacy alias for audio_encoder_dim."""
        return self.audio_encoder_dim
    
    @property
    def frame_duration_ms(self) -> float:
        """Duration of each frame in milliseconds after downsampling."""
        return 1000.0 / self.effective_frame_rate
'''

    with open(output_path / "configuration_voxlm.py", "w") as f:
        f.write(config_content)

    print(f"Created: {output_path / 'configuration_voxlm.py'}")


def convert_checkpoint(
    checkpoint_path: Path, output_path: Path, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convert model.pt checkpoint to safetensors format.

    Returns the config dict from the checkpoint.
    """
    import torch

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Extract config
    config_dict = checkpoint.get("config", {})
    if hasattr(config_dict, "__dict__"):
        # Convert dataclass to dict
        config_dict = {
            k: v for k, v in config_dict.__dict__.items() if not k.startswith("_")
        }

    # Save as safetensors
    try:
        from safetensors.torch import save_file

        # Filter out non-tensor items, clone to break shared memory, and convert to contiguous
        # This is needed because some tensors share memory (e.g., whisper.encoder == encoder, embed_tokens == lm_head)
        tensor_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                # Clone to break shared memory references
                tensor_dict[k] = v.clone().contiguous()

        save_file(tensor_dict, output_path / "model.safetensors")
        print(f"Created: {output_path / 'model.safetensors'}")
    except ImportError:
        print("Warning: safetensors not installed. Copying model.pt instead.")
        shutil.copy(checkpoint_path, output_path / "model.pt")
        print(f"Copied: {output_path / 'model.pt'}")

    return config_dict


def create_config_json(output_path: Path, config_dict: Dict[str, Any]) -> None:
    """
    Create config.json with auto_map for HuggingFace AutoModel.
    """
    # Build config.json
    config_json = {
        "model_type": "voxlm",
        "architectures": ["VoxLMForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "configuration_voxlm.VoxLMConfig",
            "AutoModel": "modeling_voxlm.VoxLMForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_voxlm.VoxLMForConditionalGeneration",
        },
        "transformers_version": "4.40.0",
    }

    # Add model config
    config_json.update(config_dict)

    with open(output_path / "config.json", "w") as f:
        json.dump(config_json, f, indent=2, default=str)

    print(f"Created: {output_path / 'config.json'}")


def copy_readme(output_path: Path) -> None:
    """Copy model README to output directory."""
    readme_path = Path(__file__).parent.parent / "models" / "voxlm-2b" / "README.md"

    if readme_path.exists():
        shutil.copy(readme_path, output_path / "README.md")
        print(f"Copied: {output_path / 'README.md'}")
    else:
        print(f"Warning: README not found at {readme_path}")


def upload_to_hf(output_path: Path, repo_id: str) -> None:
    """Upload built files to HuggingFace Hub."""
    from huggingface_hub import HfApi, upload_folder

    print(f"\nUploading to HuggingFace: {repo_id}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Upload folder
    upload_folder(
        folder_path=str(output_path),
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nUpload complete!")
    print(f"Model URL: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Build HuggingFace-compatible model files from VoxLM source"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (model.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hf/",
        help="Output directory for HF files (default: hf/)",
    )
    parser.add_argument(
        "--upload",
        type=str,
        default=None,
        help="HuggingFace repo ID to upload to (e.g., 'username/voxlm-2b')",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip weight conversion (only generate code files)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    # Validate inputs
    if not args.skip_weights and not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VoxLM HuggingFace Build")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    if args.upload:
        print(f"Upload to: {args.upload}")
    print("=" * 60)

    # 1. Merge source files
    print("\n[1/5] Merging source files...")
    merge_source_files(output_path)

    # 2. Create configuration file
    print("\n[2/5] Creating configuration file...")
    create_configuration_file(output_path)

    # 3. Convert checkpoint
    config_dict = {}
    if not args.skip_weights:
        print("\n[3/5] Converting checkpoint...")
        config_dict = convert_checkpoint(checkpoint_path, output_path)
    else:
        print("\n[3/5] Skipping weight conversion...")

    # 4. Create config.json
    print("\n[4/5] Creating config.json...")
    create_config_json(output_path, config_dict)

    # 5. Copy README
    print("\n[5/5] Copying README...")
    copy_readme(output_path)

    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"\nFiles created in: {output_path}")
    print("  - modeling_voxlm.py")
    print("  - configuration_voxlm.py")
    print("  - config.json")
    if not args.skip_weights:
        print("  - model.safetensors (or model.pt)")
    print("  - README.md")

    # Upload if requested
    if args.upload:
        print("\n" + "=" * 60)
        upload_to_hf(output_path, args.upload)

    print("\n" + "=" * 60)
    print("Usage after upload:")
    print("=" * 60)
    print(
        """
from transformers import AutoModel
import torch

# Load model
model = AutoModel.from_pretrained(
    "REPO_ID",
    trust_remote_code=True
).to("cuda")

# Transcribe
import soundfile as sf
audio, sr = sf.read("audio.wav")
result = model.generate(torch.from_numpy(audio).float().to("cuda"))
print(result["text"])
print(result["words"])  # With timestamps and confidence
""".replace("REPO_ID", args.upload or "username/voxlm-2b")
    )


if __name__ == "__main__":
    main()
