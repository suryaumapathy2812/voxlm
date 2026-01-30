"""
VoxLM: Modular Speech-to-Text with LLM Intelligence.

A flexible architecture that combines:
- Audio Encoder: Whisper, IndicWhisper, MMS, etc.
- Projection Layer: Bridges audio to LLM space
- LLM Backbone: Qwen, Llama, Phi, Gemma, etc.
- Alignment Module: For word-level timestamps
- Confidence Extraction: From LLM token probabilities

Supports any combination of encoder + LLM for different use cases:
- Global: Whisper-large-v3 + Qwen2.5-7B (99 languages)
- India: IndicWhisper + Qwen2.5-7B (12+ Indian languages)
- Edge: Whisper-tiny + Qwen2.5-0.5B (fast inference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from .config import VoxLMConfig, get_config
from .audio_encoder import AudioEncoder, AudioProjection
from .heads import TimestampHead, ConfidenceHead, AttentionBasedTimestamps
from .alignment import (
    AlignmentModule,
    TimestampExtractor,
    ConfidenceExtractor,
    alignment_loss,
    alignment_loss_from_frame_indices,
    monotonicity_loss,
    merge_timestamps_and_confidence,
    AttentionAlignmentLoss,
)


class ConfidenceStoppingCriteria(StoppingCriteria):
    """
    Stop generation when token confidence drops below threshold.

    This detects hallucination by monitoring when the model becomes
    uncertain about what to generate next (indicating it's moved beyond
    the actual audio content).
    """

    def __init__(self, threshold: float = 0.6, min_tokens: int = 2):
        """
        Args:
            threshold: Stop if confidence drops below this (default 0.6)
            min_tokens: Don't stop before generating this many tokens
        """
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.generated_count = 0
        self.stop_at_token = None
        self.confidences = []

    def reset(self):
        """Reset state for new generation."""
        self.generated_count = 0
        self.stop_at_token = None
        self.confidences = []

    def __call__(self, input_ids, scores, **kwargs):
        self.generated_count += 1

        if self.generated_count < self.min_tokens:
            return False

        if scores is None:
            return False

        # Handle tuple scores (from some generation configs)
        if isinstance(scores, tuple):
            scores = scores[-1]

        # Compute confidence for the last generated token
        probs = torch.softmax(scores, dim=-1)
        last_token = input_ids[0, -1]
        conf = probs[0, last_token].item()
        self.confidences.append(conf)

        # Stop if confidence drops below threshold
        if conf < self.threshold:
            self.stop_at_token = self.generated_count - 1  # Exclude low-conf token
            return True

        return False


@dataclass
class VoxLMOutput:
    """Output container for VoxLM model."""

    loss: Optional[torch.Tensor] = None
    text_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[Tuple] = None

    # v1 outputs (deprecated)
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None

    # v2 outputs
    alignment: Optional[torch.Tensor] = None  # [batch, text_len, audio_frames]
    audio_embeds: Optional[torch.Tensor] = None  # [batch, audio_frames, dim]
    text_start_idx: Optional[int] = None
    num_audio_frames: Optional[int] = None


class VoxLM(nn.Module):
    """
    VoxLM: Modular Speech-to-Text with LLM Intelligence.

    A flexible architecture combining any audio encoder + any LLM:
    - Audio Encoder: Whisper, IndicWhisper, MMS, Wav2Vec2, etc.
    - LLM Backbone: Qwen, Llama, Phi, Gemma, etc.
    - Alignment Module: For word-level timestamps
    - Confidence Extraction: From LLM token probabilities
    """

    def __init__(self, config: VoxLMConfig):
        super().__init__()
        self.config = config

        # 1. Audio Encoder (Whisper)
        self.audio_encoder = AudioEncoder(
            model_name=config.whisper_model,
            freeze=config.freeze_audio_encoder,
        )

        # 2. Projection Layer (audio -> LLM space)
        self.audio_projection = AudioProjection(
            audio_dim=config.whisper_dim,
            llm_dim=config.llm_dim,
            downsample_factor=config.downsample_factor,
            dropout=config.projection_dropout,
        )

        # 3. LLM Backbone (Qwen2)
        # Use bfloat16 on CUDA, float32 on CPU
        if torch.cuda.is_available():
            llm_dtype = torch.bfloat16
        else:
            llm_dtype = torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            torch_dtype=llm_dtype,
            trust_remote_code=True,
        )
        self._llm_dtype = llm_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model,
            trust_remote_code=True,
        )

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                config.audio_start_token,
                config.audio_end_token,
                config.transcribe_token,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Store special token IDs
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids(
            config.audio_start_token
        )
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids(config.audio_end_token)
        self.transcribe_id = self.tokenizer.convert_tokens_to_ids(
            config.transcribe_token
        )

        # Freeze LLM if specified
        if config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # Apply LoRA if specified
        if config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.llm = get_peft_model(self.llm, lora_config)

        # 4. Output Heads - depends on architecture version
        if config.architecture_version == "v1":
            # v1: Separate heads for timestamps and confidence (deprecated)
            self.timestamp_head = TimestampHead(
                hidden_dim=config.llm_dim,
                max_frames=config.max_audio_frames,
            )
            self.confidence_head = ConfidenceHead(
                hidden_dim=config.llm_dim,
            )
            self.attention_timestamps = AttentionBasedTimestamps()
            self.alignment_module = None
            self.timestamp_extractor = None
            self.confidence_extractor = None
        else:
            # v2: Cross-attention alignment + DTW + token probability
            self.timestamp_head = None
            self.confidence_head = None
            self.attention_timestamps = None

            # Alignment module (learnable cross-attention)
            self.alignment_module = AlignmentModule(
                hidden_dim=config.llm_dim,
                num_layers=config.alignment_num_layers,
                num_heads=config.alignment_num_heads,
                dropout=config.alignment_dropout,
            )

            # Timestamp extractor (DTW, no learnable params)
            self.timestamp_extractor = TimestampExtractor(
                frame_duration_ms=config.frame_duration_ms,
                median_filter_width=config.median_filter_width,
            )

            # Confidence extractor (uses token probability, no learnable params)
            self.confidence_extractor = ConfidenceExtractor()

        # Learnable special embeddings
        self.audio_start_embed = nn.Parameter(torch.randn(config.llm_dim) * 0.02)
        self.audio_end_embed = nn.Parameter(torch.randn(config.llm_dim) * 0.02)

        # Store dtype for casting
        self.register_buffer("_dtype_marker", torch.tensor(0, dtype=llm_dtype))

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load a pre-configured model variant."""
        config = get_config(model_name)
        return cls(config, **kwargs)

    @property
    def dtype(self):
        """Get the model's dtype."""
        return self._llm_dtype

    def get_input_embeddings(self):
        """Get LLM input embeddings layer."""
        return self.llm.get_input_embeddings()

    def encode_audio(
        self,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encode audio to embeddings in LLM space.

        Args:
            audio: Raw audio waveform [batch, samples]

        Returns:
            audio_embeds: [batch, frames, llm_dim]
            frame_mask: [batch, frames]
            num_frames: Number of audio frames
        """
        # Encode with Whisper
        audio_features, frame_mask = self.audio_encoder(audio)

        # Project to LLM space
        audio_embeds, frame_mask = self.audio_projection(audio_features, frame_mask)

        return audio_embeds, frame_mask, audio_embeds.shape[1]

    def prepare_inputs(
        self,
        audio: torch.Tensor,
        instruction: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Prepare model inputs by combining instruction and audio.

        Input format:
        [INST] {instruction} [/INST] <|audio|> {audio_embeds} <|/audio|> <|transcribe|>

        CRITICAL: Labels are NOT embedded in inputs_embeds. They're passed separately.
        This allows the model to learn to predict text, not see the answers.

        Args:
            audio: Raw audio [batch, samples]
            instruction: Optional text instruction
            labels: Optional target transcription token IDs

        Returns:
            Dict with inputs_embeds, attention_mask, labels, and PURE audio_embeds
        """
        batch_size = audio.shape[0]
        device = audio.device

        # 1. Encode audio - get PURE embeddings before they go into LLM
        audio_embeds, audio_mask, num_frames = self.encode_audio(audio)

        # CRITICAL: Store pure audio embeddings for alignment
        # These are NOT mixed with text via self-attention
        pure_audio_embeds = audio_embeds.clone()

        # 2. Encode instruction
        if instruction is None:
            instruction = "Transcribe the following audio."

        # Handle single instruction for entire batch
        if isinstance(instruction, str):
            instruction = [instruction] * batch_size

        inst_tokens = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        inst_embeds = self.get_input_embeddings()(inst_tokens.input_ids)

        # 3. Build full sequence
        # [instruction] [audio_start] [audio_frames] [audio_end] [transcribe]
        audio_start = (
            self.audio_start_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        )
        audio_end = (
            self.audio_end_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        )
        transcribe_embed = self.get_input_embeddings()(
            torch.tensor([[self.transcribe_id]], device=device).expand(batch_size, -1)
        )

        inputs_embeds = torch.cat(
            [
                inst_embeds,
                audio_start,
                audio_embeds,  # These go into LLM and get mixed via self-attention
                audio_end,
                transcribe_embed,
            ],
            dim=1,
        )

        # 4. Build attention mask (prompt only - no labels in input)
        attention_mask = torch.cat(
            [
                inst_tokens.attention_mask,
                torch.ones(batch_size, 1, device=device),  # audio_start
                audio_mask.float(),
                torch.ones(batch_size, 1, device=device),  # audio_end
                torch.ones(batch_size, 1, device=device),  # transcribe
            ],
            dim=1,
        )

        # CRITICAL: Labels are NOT added to inputs_embeds
        # They will be passed separately to LLM for loss computation

        # Track positions for later
        audio_start_idx = inst_tokens.input_ids.shape[1] + 1
        audio_end_idx = audio_start_idx + num_frames
        text_start_idx = audio_end_idx + 2  # +2 for audio_end and transcribe tokens

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_start_idx": audio_start_idx,
            "audio_end_idx": audio_end_idx,
            "text_start_idx": text_start_idx,
            "num_audio_frames": num_frames,
            # CRITICAL: Pure audio embeddings for alignment (not from LLM hidden states!)
            "pure_audio_embeds": pure_audio_embeds,
            "audio_mask": audio_mask,
        }

    def forward(
        self,
        audio: torch.Tensor,
        instruction: Optional[str] = None,
        labels: Optional[torch.Tensor] = None,
        token_timestamps: Optional[List[List[Tuple[float, float]]]] = None,
        timestamps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_alignment: bool = False,
    ) -> VoxLMOutput:
        """
        Forward pass with teacher forcing for training.

        For training:
        1. Embed labels and concatenate to inputs (teacher forcing)
        2. Compute loss on shifted labels (predict next token)

        Args:
            audio: Raw audio waveform [batch, samples]
            instruction: Optional text instruction
            labels: Target transcription token IDs [batch, seq]
            token_timestamps: v2 ground truth - per-token (start_sec, end_sec)
            timestamps: v1 ground truth - [batch, seq, 2] (start, end frame)
            return_dict: Whether to return VoxLMOutput
            return_alignment: Whether to compute alignment (v2 only)

        Returns:
            VoxLMOutput with losses, logits, etc.
        """
        # Prepare inputs (without labels)
        prepared = self.prepare_inputs(audio, instruction, None)

        # Cast to LLM dtype
        inputs_embeds = prepared["inputs_embeds"].to(self._llm_dtype)
        attention_mask = prepared["attention_mask"]

        # If labels provided, embed and concatenate for teacher forcing
        if labels is not None:
            label_embeds = self.get_input_embeddings()(labels).to(self._llm_dtype)
            inputs_embeds = torch.cat([inputs_embeds, label_embeds], dim=1)
            label_mask = (labels != self.tokenizer.pad_token_id).float()
            attention_mask = torch.cat([attention_mask, label_mask], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

        # Get hidden states
        hidden_states = outputs.hidden_states[-1]

        # Extract text portion (after prompt)
        text_start = prepared["text_start_idx"]
        text_hidden = hidden_states[:, text_start:, :]

        # Compute logits
        if hasattr(self.llm, "lm_head"):
            text_logits = self.llm.lm_head(text_hidden)
        elif hasattr(self.llm, "base_model"):
            text_logits = self.llm.base_model.model.lm_head(text_hidden)
        else:
            text_logits = None

        # Initialize outputs
        start_logits = None
        end_logits = None
        confidence = None
        alignment = None
        audio_embeds = prepared["pure_audio_embeds"].float()

        # Architecture-specific processing
        if self.config.architecture_version == "v1":
            text_hidden_f32 = text_hidden.float()
            start_logits, end_logits = self.timestamp_head(
                text_hidden_f32, num_audio_frames=prepared["num_audio_frames"]
            )
            confidence = self.confidence_head(text_hidden_f32)
        else:
            if return_alignment and labels is not None:
                # CRITICAL: audio_mask from prepare_inputs is True=valid, False=padded
                # But CrossAttentionBlock expects True=masked, False=valid
                # So we need to invert it, or pass None if no padding
                audio_mask = prepared.get("audio_mask")
                if audio_mask is not None:
                    # Invert: True (valid) -> False (not masked)
                    audio_mask = ~audio_mask
                alignment = self.alignment_module(
                    text_hidden=text_hidden.float(),
                    audio_embeds=audio_embeds,
                    audio_mask=audio_mask,
                )

        # Compute loss
        loss = None
        if labels is not None and text_logits is not None:
            label_len = labels.shape[1]
            # Shift: logits[:-1] predicts labels[1:]
            shift_logits = text_logits[:, : label_len - 1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )

            # Debug output (once)
            if not hasattr(self, "_debug_printed"):
                pred = shift_logits.argmax(dim=-1)
                print(f"\nDEBUG: First batch:")
                print(f"  Labels: {shift_labels[0, :10].tolist()}")
                print(f"  Preds:  {pred[0, :10].tolist()}")
                print(f"  Loss: {loss.item():.4f} (expected: 8-10)")
                self._debug_printed = True

            # Add alignment loss if available
            if self.config.architecture_version != "v1":
                if alignment is not None and token_timestamps is not None:
                    mask = labels != self.tokenizer.pad_token_id

                    # Debug: Print alignment info once
                    if not hasattr(self, "_align_debug_printed"):
                        print(f"\nDEBUG Alignment:")
                        print(f"  alignment shape: {alignment.shape}")
                        print(f"  num_audio_frames: {prepared['num_audio_frames']}")
                        print(f"  token_timestamps len: {len(token_timestamps)}")
                        if token_timestamps:
                            print(
                                f"  first sample timestamps len: {len(token_timestamps[0])}"
                            )
                            print(f"  labels shape: {labels.shape}")
                            print(f"  mask sum (valid tokens): {mask.sum().item()}")
                        self._align_debug_printed = True

                    align_loss = alignment_loss(
                        predicted=alignment,
                        target_timestamps=token_timestamps,
                        frame_rate=self.config.effective_frame_rate,
                        mask=mask,
                        interp_frames=self.config.alignment_interp_frames,
                    )

                    # Debug: Print alignment loss once
                    if not hasattr(self, "_align_loss_printed"):
                        print(f"  align_loss: {align_loss.item():.4f}")
                        print(f"  ce_loss: {loss.item():.4f}")
                        self._align_loss_printed = True

                    loss = loss + self.config.alignment_loss_weight * align_loss

        return VoxLMOutput(
            loss=loss,
            text_logits=text_logits,
            hidden_states=hidden_states,
            attentions=None,
            start_logits=start_logits,
            end_logits=end_logits,
            confidence=confidence,
            alignment=alignment,
            audio_embeds=audio_embeds,
            text_start_idx=text_start,
            num_audio_frames=prepared["num_audio_frames"],
        )

    @torch.no_grad()
    def transcribe(
        self,
        audio: torch.Tensor,
        instruction: Optional[str] = None,
        max_length: Optional[int] = None,
        use_attention_timestamps: bool = False,
        confidence_threshold: float = 0.6,
        use_confidence_stopping: bool = True,
    ) -> Dict:
        """
        Transcribe audio to text with timestamps.

        Args:
            audio: Raw audio waveform [samples] or [1, samples]
            instruction: Optional context instruction
            max_length: Maximum output tokens (if None, auto-calculated from audio duration)
            use_attention_timestamps: Use attention-based timestamps (v1 only)
            confidence_threshold: Stop generation if confidence drops below this (default 0.6)
            use_confidence_stopping: Whether to use confidence-based stopping (default True)

        Returns:
            Dict with 'text', 'words' (with timestamps and confidence)
        """
        self.eval()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = next(self.parameters()).device
        audio = audio.to(device)

        # Auto-calculate max_length based on audio duration as a safety fallback
        # This is a backup in case confidence stopping doesn't trigger
        audio_duration = audio.shape[-1] / self.config.sample_rate
        duration_based_max = (
            int(audio_duration * 6) + 20
        )  # More generous with confidence stopping

        if max_length is None:
            max_length = duration_based_max
        else:
            max_length = min(max_length, duration_based_max)

        max_length = max(max_length, 10)  # At least 10 tokens
        max_length = min(max_length, 448)  # Cap at 448

        # Prepare inputs
        prepared = self.prepare_inputs(audio, instruction)

        # Cast to correct dtype
        inputs_embeds = prepared["inputs_embeds"].to(self._llm_dtype)

        # Set up confidence-based stopping criteria
        confidence_stopping = None
        stopping_criteria = None
        if use_confidence_stopping:
            confidence_stopping = ConfidenceStoppingCriteria(
                threshold=confidence_threshold,
                min_tokens=2,
            )
            stopping_criteria = StoppingCriteriaList([confidence_stopping])

        # Generate text with scores for confidence
        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared["attention_mask"],
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_attentions=use_attention_timestamps
            and self.config.architecture_version == "v1",
            output_scores=True,  # Always output scores for confidence stopping
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria,
            # Prevent repetition
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
        )

        # Get generated token IDs
        generated_ids = generated.sequences[0]

        # Debug: Check what generate() returned
        print(
            f"DEBUG generate: sequences shape={generated.sequences.shape}, scores len={len(generated.scores) if generated.scores else 0}"
        )
        print(f"DEBUG generate: inputs_embeds shape={inputs_embeds.shape}")
        print(f"DEBUG generate: first 5 token ids={generated_ids[:5].tolist()}")

        # Truncate to exclude low-confidence tokens if confidence stopping triggered
        if confidence_stopping and confidence_stopping.stop_at_token is not None:
            generated_ids = generated_ids[: confidence_stopping.stop_at_token]

        # Decode text
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Get timestamps and confidence based on architecture version
        if self.config.architecture_version == "v2":
            # v2: Use alignment module + DTW + token probability
            words = self._transcribe_v2(
                audio=audio,
                instruction=instruction,
                generated=generated,
                generated_ids=generated_ids,
                prepared=prepared,
                num_tokens_used=confidence_stopping.stop_at_token
                if confidence_stopping and confidence_stopping.stop_at_token
                else None,
            )
        else:
            # v1: Use timestamp/confidence heads or attention
            words = self._transcribe_v1(
                audio=audio,
                instruction=instruction,
                generated=generated,
                generated_ids=generated_ids,
                prepared=prepared,
                use_attention_timestamps=use_attention_timestamps,
            )

        return {
            "text": text,
            "words": words,
            "audio_duration": audio.shape[-1] / self.config.sample_rate,
        }

    def _transcribe_v1(
        self,
        audio: torch.Tensor,
        instruction: Optional[str],
        generated,
        generated_ids: torch.Tensor,
        prepared: Dict,
        use_attention_timestamps: bool,
    ) -> List[Dict]:
        """v1 transcription: timestamp head or attention-based."""
        if (
            use_attention_timestamps
            and hasattr(generated, "attentions")
            and self.attention_timestamps is not None
        ):
            word_timestamps = self.attention_timestamps(
                generated.attentions,
                prepared["audio_start_idx"],
                prepared["audio_end_idx"],
                prepared["text_start_idx"],
                frame_duration_ms=1000 / self.config.effective_frame_rate,
            )[0]
        else:
            # Use timestamp head (requires another forward pass)
            outputs = self.forward(audio, instruction, return_alignment=False)

            if self.timestamp_head is not None and outputs.start_logits is not None:
                timestamps = self.timestamp_head.decode(
                    outputs.start_logits,
                    outputs.end_logits,
                    frame_duration_ms=1000 / self.config.effective_frame_rate,
                )[0]
                confidence = (
                    outputs.confidence[0].cpu().tolist()
                    if outputs.confidence is not None
                    else []
                )

                word_timestamps = []
                for i, ts in enumerate(timestamps[: len(generated_ids)]):
                    ts["confidence"] = confidence[i] if i < len(confidence) else 0.0
                    word_timestamps.append(ts)
            else:
                word_timestamps = []

        # Merge subword tokens into words
        return self._merge_subwords(generated_ids, word_timestamps)

    def _transcribe_v2(
        self,
        audio: torch.Tensor,
        instruction: Optional[str],
        generated,
        generated_ids: torch.Tensor,
        prepared: Dict,
        num_tokens_used: Optional[int] = None,
    ) -> List[Dict]:
        """v2 transcription: alignment module + DTW + token probability.

        CRITICAL: Uses PURE audio embeddings from prepare_inputs, not LLM hidden states.

        Args:
            num_tokens_used: If confidence stopping triggered, the number of tokens to use
        """
        device = audio.device

        # CRITICAL FIX: Use pure audio embeddings from prepared inputs
        # These are NOT contaminated by self-attention mixing with text
        audio_embeds = prepared["pure_audio_embeds"].float()

        # Get text embeddings for generated tokens
        # We need to embed the generated tokens and run through alignment module
        gen_ids_batch = generated_ids.unsqueeze(0)  # [1, seq_len]
        text_embeds = self.get_input_embeddings()(gen_ids_batch)

        # Run through LLM to get hidden states for generated text
        text_outputs = self.llm(
            inputs_embeds=text_embeds.to(self._llm_dtype),
            output_hidden_states=True,
            return_dict=True,
        )
        text_hidden = text_outputs.hidden_states[-1].float()

        # Compute alignment using PURE audio embeddings
        # CRITICAL: Invert audio_mask (True=valid -> True=masked for attention)
        audio_mask = prepared.get("audio_mask")
        if audio_mask is not None:
            audio_mask = ~audio_mask
        alignment = self.alignment_module(
            text_hidden=text_hidden,
            audio_embeds=audio_embeds,  # PURE, not from LLM hidden states!
            audio_mask=audio_mask,
        )

        # Calculate audio duration and actual frame count for proper timestamp extraction
        # Handle both 1D [samples] and 2D [batch, samples] audio tensors
        if audio.dim() == 1:
            audio_samples = audio.shape[0]
        else:
            audio_samples = audio.shape[-1]  # Last dimension is samples
        audio_duration = audio_samples / 16000.0  # Assuming 16kHz sample rate

        # Calculate actual number of audio frames (not padded)
        # Whisper outputs 50Hz, then we downsample by 4x to 12.5Hz
        actual_audio_frames = int(audio_duration * self.config.effective_frame_rate)

        # Truncate alignment to actual audio frames before DTW
        # This prevents DTW from using padded frames
        alignment_truncated = alignment[:, :, :actual_audio_frames]

        # Debug: print truncation info once
        if not hasattr(self, "_truncate_debug_printed"):
            print(f"\nDEBUG Alignment truncation:")
            print(f"  Original alignment shape: {alignment.shape}")
            print(f"  Audio duration: {audio_duration:.2f}s")
            print(f"  Actual audio frames: {actual_audio_frames}")
            print(f"  Truncated alignment shape: {alignment_truncated.shape}")
            self._truncate_debug_printed = True

        # Extract timestamps using DTW on truncated alignment
        word_timestamps = self.timestamp_extractor(
            alignment=alignment_truncated,
            token_ids=gen_ids_batch,
            tokenizer=self.tokenizer,
            audio_duration=audio_duration,
        )[0]  # Get first batch

        # Get confidence from generation scores
        if hasattr(generated, "scores") and generated.scores:
            # Scores are for tokens after the prompt
            scores = generated.scores

            # If confidence stopping triggered, only use scores up to that point
            if num_tokens_used is not None:
                scores = scores[:num_tokens_used]

            # CRITICAL FIX: When using inputs_embeds, generated.sequences already contains
            # ONLY the newly generated tokens (no prompt). The generated_ids passed to this
            # function is already correctly truncated if confidence stopping triggered.
            # Use generated_ids directly instead of slicing from generated.sequences.
            new_token_ids = generated_ids  # Already processed and truncated if needed

            # Debug: Check shapes match
            if len(scores) != len(new_token_ids):
                print(
                    f"DEBUG confidence: scores len={len(scores)}, token_ids len={len(new_token_ids)}"
                )
                # Align lengths - scores[i] predicts token[i]
                min_len = min(len(scores), len(new_token_ids))
                scores = scores[:min_len]
                new_token_ids = new_token_ids[:min_len]

            if len(scores) > 0 and len(new_token_ids) > 0:
                # Debug: print score shape and sample values
                print(
                    f"DEBUG: scores[0] shape={scores[0].shape}, new_token_ids shape={new_token_ids.shape}"
                )

                token_confidence = self.confidence_extractor.from_scores(
                    scores=list(scores),
                    token_ids=new_token_ids.unsqueeze(0),
                    temperature=self.config.confidence_temperature,
                )[0]  # [seq_len]

                # Debug: print confidence values
                print(
                    f"DEBUG: token_confidence shape={token_confidence.shape}, values={token_confidence[:5].tolist()}"
                )

                word_confidences = self.confidence_extractor.aggregate_to_words(
                    token_confidence=token_confidence,
                    token_ids=new_token_ids,
                    tokenizer=self.tokenizer,
                    aggregation=self.config.confidence_aggregation,
                )

                # Debug: print word confidences
                print(f"DEBUG: word_confidences={word_confidences[:3]}")

                # Merge timestamps and confidence
                word_timestamps = merge_timestamps_and_confidence(
                    word_timestamps, word_confidences
                )

        return word_timestamps

    def _merge_subwords(
        self,
        token_ids: torch.Tensor,
        token_timestamps: List[Dict],
    ) -> List[Dict]:
        """Merge subword tokens into words with combined timestamps."""
        words = []
        current_word = ""
        current_start = None
        current_end = None
        confidences = []

        for i, token_id in enumerate(token_ids):
            token = self.tokenizer.decode([token_id])

            # Skip special tokens
            if token_id in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
                continue

            ts = token_timestamps[i] if i < len(token_timestamps) else {}

            # Check if this is start of new word (starts with space or is first)
            if token.startswith(" ") or current_word == "":
                # Save previous word
                if current_word:
                    words.append(
                        {
                            "word": current_word.strip(),
                            "start": current_start,
                            "end": current_end,
                            "confidence": sum(confidences) / len(confidences)
                            if confidences
                            else 0.0,
                        }
                    )

                # Start new word
                current_word = token
                current_start = ts.get("start_ms", 0) / 1000  # Convert to seconds
                current_end = ts.get("end_ms", 0) / 1000
                confidences = [ts.get("confidence", 0.0)]
            else:
                # Continue current word
                current_word += token
                current_end = ts.get("end_ms", current_end * 1000) / 1000
                confidences.append(ts.get("confidence", 0.0))

        # Don't forget last word
        if current_word:
            words.append(
                {
                    "word": current_word.strip(),
                    "start": current_start,
                    "end": current_end,
                    "confidence": sum(confidences) / len(confidences)
                    if confidences
                    else 0.0,
                }
            )

        return words

    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable = 0
        total = 0
        for name, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
