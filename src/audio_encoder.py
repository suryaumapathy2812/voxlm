"""
Audio Encoder: Whisper-based audio feature extraction.

Uses the encoder portion of Whisper to convert audio waveforms
into frame-level features at 50Hz (20ms per frame).
"""

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor
from typing import Optional, Tuple


class AudioEncoder(nn.Module):
    """
    Whisper-based audio encoder.

    Takes raw audio waveforms and produces frame-level features.
    Output rate: 50Hz (one frame per 20ms of audio)
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        freeze: bool = True,
    ):
        super().__init__()

        # Load Whisper model (encoder only)
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = self.whisper.encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        # Get dimensions
        self.hidden_dim = self.encoder.config.d_model

        # Freeze if specified
        if freeze:
            self.freeze()

        # Store config
        self.sample_rate = 16000
        self.frame_rate = 50  # Whisper outputs 50 frames per second

    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()

    @torch.no_grad()
    def preprocess_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Convert raw audio to mel spectrogram features.

        Args:
            audio: Raw audio waveform [batch, samples] or [samples]
            sample_rate: Audio sample rate (should be 16kHz)

        Returns:
            Mel spectrogram features [batch, n_mels, time]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Convert to numpy for feature extractor
        audio_np = audio.cpu().numpy()

        # Extract mel features
        features = self.feature_extractor(
            audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        return features.input_features.to(audio.device)

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        preprocess: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to frame-level features.

        Args:
            audio: Either raw waveform [batch, samples] or
                   mel features [batch, n_mels, time]
            attention_mask: Optional mask for padded audio
            preprocess: Whether to convert raw audio to mel features

        Returns:
            hidden_states: Frame-level features [batch, frames, hidden_dim]
            frame_mask: Mask indicating valid frames [batch, frames]
        """
        # Preprocess if needed
        if preprocess and audio.dim() == 2 and audio.shape[1] > 3000:
            # Likely raw audio, not mel features
            mel_features = self.preprocess_audio(audio)
        else:
            mel_features = audio

        # Encode through Whisper encoder
        encoder_outputs = self.encoder(
            mel_features,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = encoder_outputs.last_hidden_state

        # Create frame mask (all ones if no attention mask provided)
        if attention_mask is None:
            frame_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            frame_mask = attention_mask.bool()

        return hidden_states, frame_mask

    def get_output_length(self, audio_length_samples: int) -> int:
        """
        Calculate number of output frames for given audio length.

        Whisper uses:
        - 400 sample hop length (25ms at 16kHz)
        - But outputs at 50Hz (20ms effective)

        Args:
            audio_length_samples: Number of audio samples

        Returns:
            Number of output frames
        """
        # Whisper's actual frame calculation
        return audio_length_samples // 320  # ~50Hz output

    def frames_to_time(self, frame_idx: int) -> float:
        """Convert frame index to time in seconds."""
        return frame_idx / self.frame_rate

    def time_to_frames(self, time_sec: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_sec * self.frame_rate)


class AudioProjection(nn.Module):
    """
    Projects audio encoder features to LLM embedding space.

    Optionally downsamples from 50Hz to 25Hz for efficiency.
    """

    def __init__(
        self,
        audio_dim: int,
        llm_dim: int,
        downsample_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.llm_dim = llm_dim
        self.downsample_factor = downsample_factor

        # Projection MLP
        self.projection = nn.Sequential(
            nn.Linear(audio_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
            nn.Dropout(dropout),
        )

        # Optional downsampling (50Hz -> 25Hz)
        if downsample_factor > 1:
            self.downsample = nn.Conv1d(
                llm_dim,
                llm_dim,
                kernel_size=downsample_factor,
                stride=downsample_factor,
            )
        else:
            self.downsample = None

    def forward(
        self,
        audio_features: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Project audio features to LLM space.

        Args:
            audio_features: [batch, frames, audio_dim]
            frame_mask: [batch, frames]

        Returns:
            projected: [batch, frames', llm_dim] where frames' = frames // downsample_factor
            new_mask: [batch, frames']
        """
        # Project to LLM dimension
        projected = self.projection(audio_features)

        # Downsample if needed
        if self.downsample is not None:
            # Conv1d expects [batch, channels, length]
            projected = projected.transpose(1, 2)
            projected = self.downsample(projected)
            projected = projected.transpose(1, 2)

            # Adjust mask
            if frame_mask is not None:
                # Simple downsampling: take every nth frame
                frame_mask = frame_mask[:, ::self.downsample_factor]

        return projected, frame_mask
