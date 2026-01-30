"""
Output heads for Qwen-STT.

- TimestampHead: Predicts start/end frame indices for each output token
- ConfidenceHead: Predicts confidence score for each output token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TimestampHead(nn.Module):
    """
    Predicts start and end frame indices for each output token.

    For each generated text token, predicts which audio frames
    correspond to that token's pronunciation.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_frames: int = 3000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_frames = max_frames

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate heads for start and end
        self.start_head = nn.Linear(hidden_dim // 2, max_frames)
        self.end_head = nn.Linear(hidden_dim // 2, max_frames)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_audio_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict timestamp logits.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            num_audio_frames: Actual number of audio frames (for masking)

        Returns:
            start_logits: [batch, seq_len, max_frames]
            end_logits: [batch, seq_len, max_frames]
        """
        # Shared features
        features = self.shared(hidden_states)

        # Predict start and end
        start_logits = self.start_head(features)
        end_logits = self.end_head(features)

        # Mask invalid frames if we know the actual length
        if num_audio_frames is not None and num_audio_frames < self.max_frames:
            mask = torch.ones_like(start_logits) * float('-inf')
            mask[:, :, :num_audio_frames] = 0
            start_logits = start_logits + mask
            end_logits = end_logits + mask

        return start_logits, end_logits

    def decode(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        frame_duration_ms: float = 40.0,
    ) -> list:
        """
        Decode logits to timestamp predictions.

        Args:
            start_logits: [batch, seq_len, max_frames]
            end_logits: [batch, seq_len, max_frames]
            frame_duration_ms: Duration of each frame in milliseconds

        Returns:
            List of dicts with 'start_frame', 'end_frame', 'start_ms', 'end_ms'
        """
        # Get most likely frames
        start_frames = start_logits.argmax(dim=-1)  # [batch, seq_len]
        end_frames = end_logits.argmax(dim=-1)

        # Ensure end >= start
        end_frames = torch.maximum(end_frames, start_frames)

        results = []
        for b in range(start_frames.shape[0]):
            batch_results = []
            for t in range(start_frames.shape[1]):
                sf = start_frames[b, t].item()
                ef = end_frames[b, t].item()
                batch_results.append({
                    'start_frame': sf,
                    'end_frame': ef,
                    'start_ms': sf * frame_duration_ms,
                    'end_ms': ef * frame_duration_ms,
                })
            results.append(batch_results)

        return results


class ConfidenceHead(nn.Module):
    """
    Predicts confidence score for each output token.

    Outputs a value in [0, 1] indicating model's confidence
    in the prediction.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict confidence scores.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            confidence: [batch, seq_len] in range [0, 1]
        """
        return self.head(hidden_states).squeeze(-1)


class AttentionBasedTimestamps(nn.Module):
    """
    Alternative: Extract timestamps from cross-attention weights.

    Instead of learning a separate head, use the attention weights
    between text tokens and audio frames to determine alignment.
    """

    def __init__(self, num_layers_to_use: int = 4):
        super().__init__()
        self.num_layers_to_use = num_layers_to_use

    def forward(
        self,
        attentions: Tuple[torch.Tensor, ...],
        audio_start_idx: int,
        audio_end_idx: int,
        text_start_idx: int,
        frame_duration_ms: float = 40.0,
    ) -> list:
        """
        Extract timestamps from attention weights.

        Args:
            attentions: Tuple of attention tensors from transformer
                        Each: [batch, heads, seq_len, seq_len]
            audio_start_idx: Index where audio tokens start
            audio_end_idx: Index where audio tokens end
            text_start_idx: Index where text output starts
            frame_duration_ms: Duration per frame

        Returns:
            List of timestamp dicts per token
        """
        # Average attention over last N layers and all heads
        attn_layers = attentions[-self.num_layers_to_use:]
        avg_attn = torch.stack(attn_layers).mean(dim=(0, 2))  # [batch, seq, seq]

        # Extract attention from text tokens to audio tokens
        text_to_audio = avg_attn[:, text_start_idx:, audio_start_idx:audio_end_idx]

        results = []
        for b in range(text_to_audio.shape[0]):
            batch_results = []
            for t in range(text_to_audio.shape[1]):
                attn_weights = text_to_audio[b, t]  # [num_audio_frames]

                # Find peak attention (start frame)
                peak_idx = attn_weights.argmax().item()

                # Find span using threshold (frames with attn > 0.5 * max)
                threshold = attn_weights.max() * 0.5
                active_frames = (attn_weights > threshold).nonzero(as_tuple=True)[0]

                if len(active_frames) > 0:
                    start_frame = active_frames[0].item()
                    end_frame = active_frames[-1].item()
                else:
                    start_frame = end_frame = peak_idx

                # Confidence from attention sharpness (entropy-based)
                probs = F.softmax(attn_weights, dim=0)
                entropy = -(probs * (probs + 1e-9).log()).sum()
                max_entropy = torch.log(torch.tensor(len(probs), dtype=torch.float))
                confidence = 1.0 - (entropy / max_entropy).item()

                batch_results.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_ms': start_frame * frame_duration_ms,
                    'end_ms': end_frame * frame_duration_ms,
                    'confidence': confidence,
                })
            results.append(batch_results)

        return results
