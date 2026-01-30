"""
Alignment module for Qwen-STT v2.

This module implements the correct approach to timestamp prediction:
1. Cross-attention between text and audio for alignment weights
2. Dynamic Time Warping (DTW) for optimal monotonic alignment
3. Confidence from token probability (no extra parameters)

References:
- Whisper timing.py: https://github.com/openai/whisper/blob/main/whisper/timing.py
- CrisperWhisper: https://arxiv.org/abs/2408.16589
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Try to import numba for JIT compilation (optional but recommended)
try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print(
        "Warning: numba not installed. DTW will be slower. Install with: pip install numba"
    )


# =============================================================================
# Cross-Attention Block
# =============================================================================


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention layer for audio-text alignment.

    Query: text hidden states (from LLM)
    Key/Value: audio embeddings (from projection layer)
    Output: refined text representations + attention weights for alignment
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.hidden_dim = hidden_dim

        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,  # [batch, text_len, dim] - from LLM
        key: torch.Tensor,  # [batch, audio_len, dim] - audio frames
        value: torch.Tensor,  # [batch, audio_len, dim] - audio frames
        key_padding_mask: Optional[torch.Tensor] = None,  # [batch, audio_len]
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Text hidden states [batch, text_len, hidden_dim]
            key: Audio embeddings [batch, audio_len, hidden_dim]
            value: Audio embeddings [batch, audio_len, hidden_dim]
            key_padding_mask: Mask for padded audio frames [batch, audio_len], True = masked
            return_attention: Whether to return attention weights

        Returns:
            output: Refined text representations [batch, text_len, hidden_dim]
            attn_weights: Attention weights [batch, num_heads, text_len, audio_len] if return_attention
        """
        batch_size, text_len, _ = query.shape
        audio_len = key.shape[1]

        # Project to multi-head
        q = (
            self.q_proj(query)
            .view(batch_size, text_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, audio_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, audio_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores: [batch, heads, text_len, audio_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch, audio_len] -> [batch, 1, 1, audio_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(
            attn_weights, v
        )  # [batch, heads, text_len, head_dim]

        # Reshape and project
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, text_len, self.hidden_dim)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        # Residual + LayerNorm
        x = self.norm1(query + attn_output)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        if return_attention:
            return x, attn_weights
        return x, None


# =============================================================================
# Alignment Module
# =============================================================================


class AlignmentModule(nn.Module):
    """
    Cross-attention module for audio-text alignment.

    Produces alignment weights [batch, text_len, audio_frames] that can be
    used with DTW to extract word-level timestamps.

    CRITICAL: This module takes PURE audio embeddings (from projection layer),
    NOT LLM hidden states. The audio must not be mixed with text via self-attention.

    Key features (based on Whisper + CrisperWhisper research):
    1. Cross-attention from text queries to pure audio keys/values
    2. Alignment head selection (not all heads are good for alignment)
    3. Returns raw attention weights for DTW processing
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Stack of cross-attention layers
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Project text hidden states to alignment query space
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Alignment head selection (like Whisper's model.alignment_heads)
        # Format: List of (layer_idx, head_idx) tuples
        # If None, use all heads from last layer (default behavior)
        # Set via set_alignment_heads() after evaluating on TIMIT or similar
        self._alignment_heads: Optional[List[Tuple[int, int]]] = None

        # Default: use all heads from all layers (will be refined after evaluation)
        self._use_all_heads = True

    def set_alignment_heads(self, heads: List[Tuple[int, int]]):
        """
        Set which (layer, head) pairs to use for alignment.

        This should be called after evaluating each head's alignment performance
        on a dataset with precise timestamps (e.g., TIMIT).

        Args:
            heads: List of (layer_idx, head_idx) tuples.
                   CrisperWhisper uses top 15 heads.

        Example:
            # After evaluation, select best heads:
            module.set_alignment_heads([(0, 2), (0, 5), (1, 1), (1, 3), (1, 7)])
        """
        self._alignment_heads = heads
        self._use_all_heads = False
        print(f"AlignmentModule: Using {len(heads)} selected alignment heads")

    def get_alignment_heads(self) -> Optional[List[Tuple[int, int]]]:
        """Get currently selected alignment heads."""
        return self._alignment_heads

    def forward(
        self,
        text_hidden: torch.Tensor,  # [batch, text_len, dim] - from LLM
        audio_embeds: torch.Tensor,  # [batch, audio_len, dim] - PURE audio from projection!
        audio_mask: Optional[torch.Tensor] = None,  # [batch, audio_len]
        return_all_weights: bool = False,
    ) -> torch.Tensor:
        """
        Compute alignment weights between text and audio.

        CRITICAL: audio_embeds must be the PURE audio embeddings from the projection
        layer, NOT extracted from LLM hidden states (which are contaminated by
        self-attention mixing with text).

        Args:
            text_hidden: Text hidden states from LLM [batch, text_len, hidden_dim]
            audio_embeds: PURE audio embeddings from projection [batch, audio_len, hidden_dim]
            audio_mask: Padding mask for audio [batch, audio_len], True = padded
            return_all_weights: Return all layer/head weights (for analysis/head selection)

        Returns:
            alignment: Alignment weights [batch, text_len, audio_len]
                       Higher values = stronger alignment between text token and audio frame
        """
        # Project text to query space
        queries = self.query_proj(text_hidden)

        # Collect attention weights from all layers
        # all_weights[layer] = [batch, heads, text_len, audio_len]
        all_weights = []
        x = queries

        for layer in self.layers:
            x, attn_weights = layer(
                query=x,
                key=audio_embeds,
                value=audio_embeds,
                key_padding_mask=audio_mask,
                return_attention=True,
            )
            all_weights.append(attn_weights)

        # Select alignment heads
        if self._use_all_heads or self._alignment_heads is None:
            # Default: average across all layers and heads
            # Stack: [num_layers, batch, heads, text_len, audio_len]
            stacked = torch.stack(all_weights, dim=0)
            # Average: [batch, text_len, audio_len]
            alignment = stacked.mean(dim=0).mean(dim=1)
        else:
            # Use selected heads only (like Whisper)
            selected_weights = []
            for layer_idx, head_idx in self._alignment_heads:
                if layer_idx < len(all_weights):
                    # all_weights[layer] is [batch, heads, text, audio]
                    head_weight = all_weights[layer_idx][:, head_idx, :, :]
                    selected_weights.append(head_weight)

            if selected_weights:
                # Stack and average: [batch, text_len, audio_len]
                stacked = torch.stack(
                    selected_weights, dim=1
                )  # [batch, num_selected, text, audio]
                alignment = stacked.mean(dim=1)
            else:
                # Fallback if no valid heads
                stacked = torch.stack(all_weights, dim=0)
                alignment = stacked.mean(dim=0).mean(dim=1)

        if return_all_weights:
            return alignment, all_weights

        return alignment


# =============================================================================
# DTW Implementation
# =============================================================================

if HAS_NUMBA:

    @numba.jit(nopython=True)
    def _dtw_backtrace(trace: np.ndarray) -> np.ndarray:
        """Backtrace through DTW cost matrix to find alignment path."""
        i = trace.shape[0] - 1
        j = trace.shape[1] - 1

        # Initialize boundaries
        trace[0, :] = 2  # horizontal moves at top row
        trace[:, 0] = 1  # vertical moves at left column

        result = []
        while i > 0 or j > 0:
            result.append((i - 1, j - 1))
            if trace[i, j] == 0:
                i -= 1
                j -= 1
            elif trace[i, j] == 1:
                i -= 1
            elif trace[i, j] == 2:
                j -= 1

        # Convert to numpy array and reverse (numba doesn't support reversed())
        n = len(result)
        path = np.empty((2, n), dtype=np.int64)
        for idx in range(n):
            # Reverse by reading from end
            ti, tj = result[n - 1 - idx]
            path[0, idx] = ti
            path[1, idx] = tj

        return path

    @numba.jit(nopython=True)
    def _dtw_core(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamic Time Warping with monotonicity constraint.

        Args:
            cost_matrix: [text_len, audio_frames] - lower = better match

        Returns:
            text_indices, time_indices: aligned index pairs
        """
        N, M = cost_matrix.shape

        # DP tables
        dp = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
        trace = np.full((N + 1, M + 1), -1, dtype=np.int32)

        dp[0, 0] = 0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # Three possible moves: diagonal, vertical, horizontal
                c0 = dp[i - 1, j - 1]  # diagonal (match)
                c1 = dp[i - 1, j]  # vertical (skip audio frame)
                c2 = dp[i, j - 1]  # horizontal (repeat text token)

                if c0 <= c1 and c0 <= c2:
                    dp[i, j] = cost_matrix[i - 1, j - 1] + c0
                    trace[i, j] = 0
                elif c1 <= c2:
                    dp[i, j] = cost_matrix[i - 1, j - 1] + c1
                    trace[i, j] = 1
                else:
                    dp[i, j] = cost_matrix[i - 1, j - 1] + c2
                    trace[i, j] = 2

        path = _dtw_backtrace(trace)
        return path[0], path[1]

else:
    # Pure Python fallback (slower)
    def _dtw_core(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pure Python DTW implementation (fallback when numba not available)."""
        N, M = cost_matrix.shape

        # DP tables
        dp = np.full((N + 1, M + 1), np.inf, dtype=np.float32)
        trace = np.full((N + 1, M + 1), -1, dtype=np.int32)

        dp[0, 0] = 0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                costs = [dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]]
                min_idx = np.argmin(costs)
                dp[i, j] = cost_matrix[i - 1, j - 1] + costs[min_idx]
                trace[i, j] = min_idx

        # Backtrace
        i, j = N, M
        trace[0, :] = 2
        trace[:, 0] = 1

        path = []
        while i > 0 or j > 0:
            path.append((i - 1, j - 1))
            if trace[i, j] == 0:
                i -= 1
                j -= 1
            elif trace[i, j] == 1:
                i -= 1
            else:
                j -= 1

        path = list(reversed(path))
        text_indices = np.array([p[0] for p in path], dtype=np.int64)
        time_indices = np.array([p[1] for p in path], dtype=np.int64)

        return text_indices, time_indices


def dtw_alignment(
    alignment_weights: torch.Tensor,  # [text_len, audio_frames]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run DTW on alignment weights to get optimal monotonic path.

    Args:
        alignment_weights: Attention weights [text_len, audio_frames]

    Returns:
        text_indices: Token indices along the path
        time_indices: Frame indices along the path
    """
    # Convert to cost matrix (negate since DTW minimizes)
    # Higher attention = lower cost
    weights = alignment_weights.detach().cpu().numpy()

    # Normalize for numerical stability
    std = weights.std()
    mean = weights.mean()
    if std > 0:
        weights = (weights - mean) / std

    # Negate to convert similarity to cost
    cost_matrix = -weights.astype(np.float32)

    return _dtw_core(cost_matrix)


# =============================================================================
# Timestamp Extractor
# =============================================================================


@dataclass
class WordTimestamp:
    """Word with timestamp and confidence."""

    word: str
    start: float  # seconds
    end: float  # seconds
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
        }


class TimestampExtractor:
    """
    Extract word-level timestamps from alignment weights using DTW.

    This is the same approach Whisper uses (see whisper/timing.py):
    1. Apply median filter to alignment weights for smoothness
    2. Run DTW to find optimal monotonic alignment path
    3. Map path to word boundaries
    """

    def __init__(
        self,
        frame_duration_ms: float = 80.0,  # 12.5Hz = 80ms per frame
        median_filter_width: int = 7,
    ):
        self.frame_duration_ms = frame_duration_ms
        self.median_filter_width = median_filter_width

    def __call__(
        self,
        alignment: torch.Tensor,  # [batch, text_len, audio_frames]
        token_ids: torch.Tensor,  # [batch, text_len]
        tokenizer,
        audio_duration: float = None,  # Optional: actual audio duration for clamping
    ) -> List[List[Dict]]:
        """
        Extract word timestamps from alignment weights.

        Args:
            alignment: Alignment weights [batch, text_len, audio_frames]
            token_ids: Generated token IDs [batch, text_len]
            tokenizer: Tokenizer for decoding tokens
            audio_duration: Optional audio duration in seconds for clamping

        Returns:
            List of word dicts per batch: [[{"word": str, "start": float, "end": float}, ...], ...]
        """
        batch_results = []

        # Debug: print alignment shape once
        if not hasattr(self, "_debug_printed"):
            print(f"\nDEBUG TimestampExtractor:")
            print(f"  alignment shape: {alignment.shape}")
            print(f"  token_ids shape: {token_ids.shape}")
            print(f"  frame_duration_ms: {self.frame_duration_ms}")
            num_frames = alignment.shape[2]
            max_time = (num_frames * self.frame_duration_ms) / 1000.0
            print(f"  num_audio_frames: {num_frames}")
            print(f"  max possible time: {max_time:.2f}s")
            self._debug_printed = True

        for b in range(alignment.shape[0]):
            weights = alignment[b]  # [text_len, audio_frames]

            # 1. Apply median filter for smoothness
            weights = self._median_filter(weights)

            # 2. Run DTW
            text_indices, time_indices = dtw_alignment(weights)

            # Debug: print DTW output once
            if not hasattr(self, "_dtw_debug_printed"):
                print(
                    f"  DTW text_indices range: {text_indices.min()} - {text_indices.max()}"
                )
                print(
                    f"  DTW time_indices range: {time_indices.min()} - {time_indices.max()}"
                )
                self._dtw_debug_printed = True

            # 3. Convert to word timestamps
            words = self._indices_to_words(
                text_indices,
                time_indices,
                token_ids[b],
                tokenizer,
            )

            # 4. Clamp timestamps to audio duration if provided
            if audio_duration is not None:
                for word in words:
                    word["start"] = min(max(word["start"], 0), audio_duration)
                    word["end"] = min(max(word["end"], 0), audio_duration)
                    if word["end"] < word["start"]:
                        word["end"] = word["start"]

            batch_results.append(words)

        return batch_results

    def _median_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply median filter along time dimension for smoothness."""
        if self.median_filter_width <= 1:
            return x

        # Pad and unfold for median computation
        pad = self.median_filter_width // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")

        # Unfold to get sliding windows
        x_unfolded = x_padded.unfold(-1, self.median_filter_width, 1)

        # Compute median
        return x_unfolded.median(dim=-1).values

    def _indices_to_words(
        self,
        text_indices: np.ndarray,
        time_indices: np.ndarray,
        token_ids: torch.Tensor,
        tokenizer,
    ) -> List[Dict]:
        """Convert DTW path to word-level timestamps."""

        # Find where text index changes (token boundaries)
        jumps = np.diff(text_indices, prepend=-1) > 0

        # Get frame index for each token
        token_frames = {}
        for text_idx, time_idx in zip(text_indices, time_indices):
            if text_idx not in token_frames:
                token_frames[text_idx] = {"start": time_idx, "end": time_idx}
            else:
                token_frames[text_idx]["end"] = time_idx

        # Build words from tokens
        words = []
        current_word = ""
        current_start = None
        current_end = None

        # Get special token IDs to skip
        special_ids = set()
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            special_ids.add(tokenizer.pad_token_id)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            special_ids.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            special_ids.add(tokenizer.bos_token_id)

        for i, token_id in enumerate(token_ids):
            token_id_val = token_id.item()

            # Skip special tokens
            if token_id_val in special_ids:
                continue

            # Decode token
            token_text = tokenizer.decode([token_id_val])

            # Get frame info for this token
            frame_info = token_frames.get(i, {"start": 0, "end": 0})
            start_time = (frame_info["start"] * self.frame_duration_ms) / 1000.0
            end_time = (frame_info["end"] * self.frame_duration_ms) / 1000.0

            # Check if this starts a new word (starts with space or is first)
            if (
                token_text.startswith(" ")
                or token_text.startswith("\n")
                or current_word == ""
            ):
                # Save previous word
                if current_word.strip():
                    words.append(
                        {
                            "word": current_word.strip(),
                            "start": round(current_start, 3),
                            "end": round(current_end, 3),
                        }
                    )

                # Start new word
                current_word = token_text
                current_start = start_time
                current_end = end_time
            else:
                # Continue current word
                current_word += token_text
                current_end = end_time

        # Don't forget last word
        if current_word.strip():
            words.append(
                {
                    "word": current_word.strip(),
                    "start": round(current_start, 3),
                    "end": round(current_end, 3),
                }
            )

        return words


# =============================================================================
# Confidence Extractor
# =============================================================================


class ConfidenceExtractor:
    """
    Extract confidence scores from LLM token probabilities.

    No extra parameters needed - LLM probabilities are already well-calibrated.
    This replaces the flawed ConfidenceHead from v1 which had no training signal.
    """

    @staticmethod
    def from_logits(
        logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        token_ids: torch.Tensor,  # [batch, seq_len]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Get confidence as probability of generated token.

        Args:
            logits: LLM output logits [batch, seq_len, vocab_size]
            token_ids: Generated token IDs [batch, seq_len]
            temperature: Temperature for probability scaling (1.0 = no scaling)

        Returns:
            confidence: [batch, seq_len] in range [0, 1]
        """
        # Apply temperature scaling
        scaled_logits = logits / max(temperature, 1e-8)

        # Softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Get probability of the actual generated token
        # token_ids: [batch, seq_len] -> [batch, seq_len, 1]
        confidence = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

        # Debug: check for issues
        if confidence.max() < 0.01:
            print(
                f"DEBUG from_logits: Very low confidence! max={confidence.max().item():.4f}"
            )
            print(
                f"  logits shape={logits.shape}, range=[{logits.min().item():.2f}, {logits.max().item():.2f}]"
            )
            print(
                f"  probs max per position: {probs.max(dim=-1).values[0, :5].tolist()}"
            )
            print(f"  token_ids sample: {token_ids[0, :5].tolist()}")

        return confidence

    @staticmethod
    def from_scores(
        scores: List[torch.Tensor],  # List of [batch, vocab_size] from generate()
        token_ids: torch.Tensor,  # [batch, seq_len]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Get confidence from generation scores (output of model.generate()).

        Args:
            scores: List of score tensors from generate(output_scores=True)
            token_ids: Generated token IDs [batch, seq_len]
            temperature: Temperature for probability scaling

        Returns:
            confidence: [batch, seq_len] in range [0, 1]
        """
        if not scores:
            return torch.zeros_like(token_ids, dtype=torch.float)

        # Stack scores: list of [batch, vocab_size] -> [batch, seq_len, vocab_size]
        # torch.stack with dim=1 inserts new dim at position 1
        stacked = torch.stack(scores, dim=1)

        # Debug: verify shapes
        batch_size, seq_len = token_ids.shape
        if stacked.shape[1] != seq_len:
            print(
                f"DEBUG from_scores: stacked shape={stacked.shape}, token_ids shape={token_ids.shape}"
            )
            # Truncate to match
            min_len = min(stacked.shape[1], seq_len)
            stacked = stacked[:, :min_len, :]
            token_ids = token_ids[:, :min_len]

        return ConfidenceExtractor.from_logits(stacked, token_ids, temperature)

    @staticmethod
    def aggregate_to_words(
        token_confidence: torch.Tensor,  # [seq_len]
        token_ids: torch.Tensor,  # [seq_len]
        tokenizer,
        aggregation: str = "min",  # "min", "mean", or "product"
    ) -> List[Dict]:
        """
        Aggregate subword confidence to word-level.

        Args:
            token_confidence: Per-token confidence [seq_len]
            token_ids: Token IDs [seq_len]
            tokenizer: Tokenizer for decoding
            aggregation: How to combine subword confidences
                - "min": Most uncertain subword determines word confidence (conservative)
                - "mean": Average confidence across subwords
                - "product": Multiply probabilities (very strict)

        Returns:
            List of dicts: [{"word": str, "confidence": float}, ...]
        """
        words = []
        current_word = ""
        current_confidences = []

        # Get special token IDs
        special_ids = set()
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            special_ids.add(tokenizer.pad_token_id)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            special_ids.add(tokenizer.eos_token_id)

        for i, token_id in enumerate(token_ids):
            token_id_val = token_id.item()

            if token_id_val in special_ids:
                continue

            token_text = tokenizer.decode([token_id_val])
            conf = token_confidence[i].item() if i < len(token_confidence) else 0.0

            # New word starts with space
            if (
                token_text.startswith(" ")
                or token_text.startswith("\n")
                or current_word == ""
            ):
                # Save previous word
                if current_word.strip() and current_confidences:
                    word_conf = ConfidenceExtractor._aggregate(
                        current_confidences, aggregation
                    )
                    words.append(
                        {
                            "word": current_word.strip(),
                            "confidence": round(word_conf, 3),
                        }
                    )

                current_word = token_text
                current_confidences = [conf]
            else:
                current_word += token_text
                current_confidences.append(conf)

        # Last word
        if current_word.strip() and current_confidences:
            word_conf = ConfidenceExtractor._aggregate(current_confidences, aggregation)
            words.append(
                {
                    "word": current_word.strip(),
                    "confidence": round(word_conf, 3),
                }
            )

        return words

    @staticmethod
    def _aggregate(confidences: List[float], method: str) -> float:
        """Aggregate confidence values."""
        if not confidences:
            return 0.0

        if method == "min":
            return min(confidences)
        elif method == "mean":
            return sum(confidences) / len(confidences)
        elif method == "product":
            result = 1.0
            for c in confidences:
                result *= c
            return result
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


# =============================================================================
# Alignment Loss
# =============================================================================


class AttentionAlignmentLoss(nn.Module):
    """
    CrisperWhisper-style attention supervision loss.

    This is the KEY innovation from CrisperWhisper (not in their paper!):
    - Ground truth: L2-normalized vector where 1 = word active, 0 = inactive
    - Linear interpolation at boundaries for smoothness
    - Loss: 1 - cosine_similarity(predicted, ground_truth)

    Reference: https://github.com/nyrahealth/CrisperWhisper README
    """

    def __init__(
        self,
        frame_rate: float = 12.5,  # Hz (frames per second)
        interp_frames: int = 4,  # Linear interpolation width at boundaries (8ms at 50Hz)
        clip_range: float = 4.0,  # Clip attention outside ±N seconds of ground truth
    ):
        super().__init__()
        self.frame_rate = frame_rate
        self.interp_frames = interp_frames
        self.clip_range_frames = int(clip_range * frame_rate)

    def forward(
        self,
        predicted_attn: torch.Tensor,  # [batch, text_len, audio_frames]
        token_timestamps: List[
            List[Tuple[float, float]]
        ],  # Per-token (start_sec, end_sec)
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # [batch, text_len] - which tokens are real
    ) -> torch.Tensor:
        """
        Compute attention supervision loss.

        Args:
            predicted_attn: Predicted attention weights [batch, text_len, audio_frames]
            token_timestamps: List of lists of (start_sec, end_sec) tuples for each token
                             token_timestamps[batch_idx][token_idx] = (start, end)
            attention_mask: Mask for valid tokens [batch, text_len], 1 = valid

        Returns:
            loss: Scalar loss value (1 - mean cosine similarity)
        """
        batch_size, text_len, num_frames = predicted_attn.shape
        device = predicted_attn.device

        # Create ground truth attention
        gt_attn = torch.zeros_like(predicted_attn)

        for b in range(batch_size):
            if b >= len(token_timestamps):
                continue

            for t, (start_sec, end_sec) in enumerate(token_timestamps[b]):
                if t >= text_len:
                    break

                # Convert seconds to frames
                start_frame = int(start_sec * self.frame_rate)
                end_frame = int(end_sec * self.frame_rate)

                # Clamp to valid range
                start_frame = max(0, min(start_frame, num_frames - 1))
                end_frame = max(start_frame + 1, min(end_frame + 1, num_frames))

                # Set active region to 1
                gt_attn[b, t, start_frame:end_frame] = 1.0

                # Linear interpolation at boundaries (CrisperWhisper uses 4 steps = 8ms)
                for i in range(self.interp_frames):
                    alpha = (i + 1) / (self.interp_frames + 1)

                    # Ramp up before start
                    idx = start_frame - self.interp_frames + i
                    if 0 <= idx < num_frames:
                        gt_attn[b, t, idx] = alpha

                    # Ramp down after end
                    idx = end_frame + i
                    if 0 <= idx < num_frames:
                        gt_attn[b, t, idx] = 1.0 - alpha

        # L2 normalize ground truth (as per CrisperWhisper)
        gt_norm = torch.norm(gt_attn, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        gt_attn_normalized = gt_attn / gt_norm

        # L2 normalize predictions
        pred_norm = torch.norm(predicted_attn, p=2, dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        pred_attn_normalized = predicted_attn / pred_norm

        # Cosine similarity: dot product of normalized vectors
        cos_sim = (pred_attn_normalized * gt_attn_normalized).sum(
            dim=-1
        )  # [batch, text_len]

        # Loss: 1 - cosine_similarity
        loss_per_token = 1.0 - cos_sim

        # Apply attention mask if provided
        if attention_mask is not None:
            loss_per_token = loss_per_token * attention_mask.float()
            loss = loss_per_token.sum() / (attention_mask.sum().clamp(min=1))
        else:
            loss = loss_per_token.mean()

        return loss


def alignment_loss(
    predicted: torch.Tensor,  # [batch, text_len, audio_frames]
    target_timestamps: List[
        List[Tuple[float, float]]
    ],  # Per-token (start_sec, end_sec)
    frame_rate: float = 12.5,
    mask: Optional[torch.Tensor] = None,  # [batch, text_len] - 1 = valid token
    interp_frames: int = 4,
) -> torch.Tensor:
    """
    CrisperWhisper-style attention supervision loss (functional interface).

    This is the correct loss function based on deep research:
    - Ground truth: L2-normalized vector, 1 where word active, 0 elsewhere
    - Linear interpolation at boundaries for smoothness
    - Loss: 1 - cosine_similarity(predicted, ground_truth)

    Args:
        predicted: Predicted alignment weights [batch, text_len, audio_frames]
        target_timestamps: List of lists of (start_sec, end_sec) for each token
        frame_rate: Audio frame rate in Hz (default 12.5 for 80ms frames)
        mask: Mask for valid tokens [batch, text_len]
        interp_frames: Number of frames for linear interpolation at boundaries

    Returns:
        loss: Scalar loss value
    """
    loss_fn = AttentionAlignmentLoss(
        frame_rate=frame_rate,
        interp_frames=interp_frames,
    )
    return loss_fn(predicted, target_timestamps, mask)


def alignment_loss_from_frame_indices(
    predicted: torch.Tensor,  # [batch, text_len, audio_frames]
    target_frames: torch.Tensor,  # [batch, text_len] - center frame index per token
    frame_rate: float = 12.5,
    sigma: float = 2.0,  # Gaussian width in frames
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Alternative: Gaussian-based alignment loss using frame indices.

    Use this when you only have center frame indices (not start/end timestamps).
    Creates soft Gaussian targets around the correct frame.

    Args:
        predicted: Predicted alignment weights [batch, text_len, audio_frames]
        target_frames: Ground truth center frame index for each token [batch, text_len]
        frame_rate: Audio frame rate in Hz
        sigma: Width of Gaussian target (in frames)
        mask: Mask for valid tokens [batch, text_len]

    Returns:
        loss: Scalar loss value
    """
    device = predicted.device
    batch_size, text_len, num_frames = predicted.shape

    # Create frame indices: [audio_frames]
    frame_idx = torch.arange(num_frames, device=device, dtype=torch.float32)

    # Expand target_frames: [batch, text_len, 1]
    target_expanded = target_frames.unsqueeze(-1).float()

    # Compute Gaussian target distribution
    target_dist = torch.exp(-((frame_idx - target_expanded) ** 2) / (2 * sigma**2))

    # L2 normalize (like CrisperWhisper)
    target_norm = torch.norm(target_dist, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    target_normalized = target_dist / target_norm

    pred_norm = torch.norm(predicted, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    pred_normalized = predicted / pred_norm

    # Cosine similarity loss
    cos_sim = (pred_normalized * target_normalized).sum(dim=-1)
    loss_per_token = 1.0 - cos_sim

    # Apply mask if provided
    if mask is not None:
        loss_per_token = loss_per_token * mask.float()
        loss = loss_per_token.sum() / (mask.sum().clamp(min=1))
    else:
        loss = loss_per_token.mean()

    return loss


def monotonicity_loss(
    predicted: torch.Tensor,  # [batch, text_len, audio_frames]
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Monotonicity regularization loss.

    Encourages the alignment to be monotonic (later tokens align to later frames).
    This is a soft constraint that can be added to the main alignment loss.

    Args:
        predicted: Predicted alignment weights [batch, text_len, audio_frames]
        mask: Mask for valid tokens [batch, text_len]

    Returns:
        loss: Monotonicity penalty
    """
    device = predicted.device
    num_frames = predicted.shape[-1]

    # Get expected frame position for each token
    frame_indices = torch.arange(num_frames, device=device, dtype=torch.float32)
    expected_frames = (predicted * frame_indices).sum(dim=-1)  # [batch, text_len]

    # Compute differences between consecutive tokens
    # Should be positive (monotonically increasing)
    diffs = expected_frames[:, 1:] - expected_frames[:, :-1]

    # Penalize negative differences (non-monotonic)
    violations = F.relu(-diffs)

    if mask is not None:
        # Mask for consecutive pairs
        pair_mask = mask[:, 1:] * mask[:, :-1]
        violations = violations * pair_mask.float()
        loss = violations.sum() / (pair_mask.sum().clamp(min=1))
    else:
        loss = violations.mean()

    return loss


# =============================================================================
# Utility Functions
# =============================================================================


def merge_timestamps_and_confidence(
    word_timestamps: List[Dict],
    word_confidences: List[Dict],
) -> List[Dict]:
    """
    Merge timestamp and confidence information for words.

    Args:
        word_timestamps: List of {"word": str, "start": float, "end": float}
        word_confidences: List of {"word": str, "confidence": float}

    Returns:
        Merged list with all fields
    """
    # Build confidence lookup
    conf_lookup = {w["word"]: w["confidence"] for w in word_confidences}

    # Merge
    result = []
    for i, ts in enumerate(word_timestamps):
        merged = ts.copy()

        # Try to find matching confidence
        if i < len(word_confidences):
            merged["confidence"] = word_confidences[i]["confidence"]
        elif ts["word"] in conf_lookup:
            merged["confidence"] = conf_lookup[ts["word"]]
        else:
            merged["confidence"] = 0.0

        result.append(merged)

    return result
