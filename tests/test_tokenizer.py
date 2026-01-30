"""
Tests for tokenizer setup - Critical for EOS learning.

These tests ensure the model will learn to generate EOS during training.
See docs/LESSONS_LEARNED.md for context on why this matters.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenizerSetup:
    """
    Test that tokenizer is configured correctly for EOS learning.

    CRITICAL: If pad_token_id == eos_token_id, the model will NEVER
    learn to generate EOS because it's ignored in the loss computation.
    """

    def test_pad_token_different_from_eos(self):
        """
        Verify pad_token_id != eos_token_id.

        This is the most important test in the entire codebase.
        If this fails, the model will hallucinate forever.
        """
        from src.model import VoxLM
        from src.config import get_config

        model = VoxLM(get_config("voxlm-2b"))

        assert model.tokenizer.pad_token_id is not None, (
            "pad_token_id should not be None"
        )

        assert model.tokenizer.eos_token_id is not None, (
            "eos_token_id should not be None"
        )

        assert model.tokenizer.pad_token_id != model.tokenizer.eos_token_id, (
            f"CRITICAL: pad_token_id ({model.tokenizer.pad_token_id}) must differ from "
            f"eos_token_id ({model.tokenizer.eos_token_id}) for EOS learning! "
            "See docs/LESSONS_LEARNED.md for details."
        )

    def test_eos_token_exists(self):
        """Verify EOS token is defined."""
        from src.model import VoxLM
        from src.config import get_config

        model = VoxLM(get_config("voxlm-2b"))

        assert model.tokenizer.eos_token is not None, "eos_token should be defined"
        assert model.tokenizer.eos_token_id is not None, (
            "eos_token_id should be defined"
        )

    def test_pad_token_exists(self):
        """Verify PAD token is defined."""
        from src.model import VoxLM
        from src.config import get_config

        model = VoxLM(get_config("voxlm-2b"))

        assert model.tokenizer.pad_token is not None, "pad_token should be defined"
        assert model.tokenizer.pad_token_id is not None, (
            "pad_token_id should be defined"
        )

    def test_special_audio_tokens_exist(self):
        """Verify audio special tokens are added."""
        from src.model import VoxLM
        from src.config import get_config

        config = get_config("voxlm-2b")
        model = VoxLM(config)

        # Check audio tokens exist
        audio_start_id = model.tokenizer.convert_tokens_to_ids(config.audio_start_token)
        audio_end_id = model.tokenizer.convert_tokens_to_ids(config.audio_end_token)
        transcribe_id = model.tokenizer.convert_tokens_to_ids(config.transcribe_token)

        assert audio_start_id is not None, (
            f"audio_start_token '{config.audio_start_token}' not in vocabulary"
        )
        assert audio_end_id is not None, (
            f"audio_end_token '{config.audio_end_token}' not in vocabulary"
        )
        assert transcribe_id is not None, (
            f"transcribe_token '{config.transcribe_token}' not in vocabulary"
        )


class TestTrainingLabels:
    """Test that training data is prepared correctly."""

    def test_eos_in_collate_output(self):
        """Verify CollateFn adds EOS to labels."""
        from scripts.train import CollateFn
        from src.model import VoxLM
        from src.config import get_config

        model = VoxLM(get_config("voxlm-2b"))
        collate_fn = CollateFn(model.tokenizer, max_text_length=448, add_eos=True)

        # Create dummy batch
        batch = [
            {"audio": __import__("torch").randn(16000), "text": "hello world"},
            {"audio": __import__("torch").randn(16000), "text": "test sentence"},
        ]

        result = collate_fn(batch)
        labels = result["labels"]

        # Check that EOS is in the labels
        eos_id = model.tokenizer.eos_token_id
        for i, label_seq in enumerate(labels):
            assert eos_id in label_seq.tolist(), (
                f"EOS token not found in labels for sample {i}. "
                "Training labels must include EOS for the model to learn when to stop."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
