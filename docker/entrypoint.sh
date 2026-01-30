#!/bin/bash
# =============================================================================
# VoxLM Docker Entrypoint
# =============================================================================
# Orchestrates the complete training pipeline inside Docker.
#
# Usage (from docker run):
#   docker run --gpus all voxlm:latest                    # Default config
#   docker run --gpus all voxlm:latest configs/voxlm-2b.yaml  # Specific config
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  VoxLM Docker Training${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# =============================================================================
# Configuration
# =============================================================================

# Config file from command line or environment
CONFIG_FILE="${1:-${VOXLM_CONFIG:-configs/voxlm-2b.yaml}}"

# Directories (from environment or defaults)
DATA_DIR="${VOXLM_DATA_DIR:-/app/data}"
OUTPUT_DIR="${VOXLM_OUTPUT_DIR:-/app/output}"
CHECKPOINT_DIR="${VOXLM_CHECKPOINT_DIR:-/app/checkpoints}"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Data dir: $DATA_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo ""

# =============================================================================
# Validate Environment
# =============================================================================

echo -e "${BLUE}[1/6]${NC} Validating environment..."

# Check CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')")
    echo -e "${GREEN}[OK]${NC} GPU: $GPU_NAME ($GPU_MEM)"
else
    echo -e "${YELLOW}[WARN]${NC} No GPU detected, training will be slow"
fi

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} Config file not found: $CONFIG_FILE"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Config file exists"

# =============================================================================
# Parse Configuration
# =============================================================================

echo -e "${BLUE}[2/6]${NC} Parsing configuration..."

# Parse YAML value using Python (handles nested YAML properly)
parse_yaml() {
    local file=$1
    local key=$2
    local default=$3
    
    local value=$(python -c "
import yaml
def get_nested(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d
with open('$file') as f:
    config = yaml.safe_load(f)
keys = '$key'.split('.')
value = get_nested(config, keys)
if value is not None:
    print(value)
" 2>/dev/null)
    
    if [ -z "$value" ]; then
        echo "$default"
    else
        echo "$value"
    fi
}

MODEL_NAME=$(parse_yaml "$CONFIG_FILE" "model.name" "voxlm-2b")
TRAIN_SPLIT=$(parse_yaml "$CONFIG_FILE" "data.train_split" "train-clean-100")
VAL_SPLIT=$(parse_yaml "$CONFIG_FILE" "data.val_split" "dev-clean")
WHISPER_MODEL=$(parse_yaml "$CONFIG_FILE" "data.whisper_model" "small")

echo "  Model: $MODEL_NAME"
echo "  Train split: $TRAIN_SPLIT"
echo "  Val split: $VAL_SPLIT"

# =============================================================================
# Check/Download Data
# =============================================================================

echo -e "${BLUE}[3/6]${NC} Checking data..."

TRAIN_DATA="$DATA_DIR/LibriSpeech/$TRAIN_SPLIT"
VAL_DATA="$DATA_DIR/LibriSpeech/$VAL_SPLIT"

if [ -d "$TRAIN_DATA" ] && [ -d "$VAL_DATA" ]; then
    echo -e "${GREEN}[OK]${NC} Data exists"
else
    echo -e "${YELLOW}[WARN]${NC} Data not found"
    echo "Please mount your data directory:"
    echo "  docker run -v /path/to/data:/app/data ..."
    echo ""
    echo "Or download LibriSpeech manually:"
    echo "  wget https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    echo "  wget https://www.openslr.org/resources/12/dev-clean.tar.gz"
    exit 1
fi

# =============================================================================
# Generate Timestamps
# =============================================================================

echo -e "${BLUE}[4/6]${NC} Checking timestamps..."

TIMESTAMPS_DIR="$DATA_DIR/timestamps"
TRAIN_TS="$TIMESTAMPS_DIR/$TRAIN_SPLIT/timestamps_manifest.json"
VAL_TS="$TIMESTAMPS_DIR/$VAL_SPLIT/timestamps_manifest.json"

if [ -f "$TRAIN_TS" ] && [ -f "$VAL_TS" ]; then
    echo -e "${GREEN}[OK]${NC} Timestamps exist"
else
    echo "Generating timestamps with Whisper $WHISPER_MODEL..."
    
    if [ ! -f "$TRAIN_TS" ]; then
        python scripts/generate_timestamps.py \
            --data "$DATA_DIR" \
            --split "$TRAIN_SPLIT" \
            --output "$TIMESTAMPS_DIR" \
            --model "$WHISPER_MODEL" \
            --fast
    fi
    
    if [ ! -f "$VAL_TS" ]; then
        python scripts/generate_timestamps.py \
            --data "$DATA_DIR" \
            --split "$VAL_SPLIT" \
            --output "$TIMESTAMPS_DIR" \
            --model "$WHISPER_MODEL" \
            --fast
    fi
    
    echo -e "${GREEN}[OK]${NC} Timestamps generated"
fi

# =============================================================================
# Phase 1: Basic Transcription Training
# =============================================================================

echo -e "${BLUE}[5/6]${NC} Phase 1: Training basic transcription..."

PHASE1_CKPT="$CHECKPOINT_DIR/$MODEL_NAME/best.pt"

if [ -f "$PHASE1_CKPT" ]; then
    echo -e "${YELLOW}[SKIP]${NC} Phase 1 checkpoint exists"
else
    python scripts/train.py \
        --model "$MODEL_NAME" \
        --data "$DATA_DIR" \
        --output "$CHECKPOINT_DIR" \
        --train-split "$TRAIN_SPLIT" \
        --val-split "$VAL_SPLIT"
    
    echo -e "${GREEN}[OK]${NC} Phase 1 complete"
fi

# =============================================================================
# Phase 2: Timestamp Alignment Training
# =============================================================================

echo -e "${BLUE}[6/6]${NC} Phase 2: Training timestamp alignment..."

PHASE2_CKPT="$CHECKPOINT_DIR/$MODEL_NAME/best_phase2.pt"

if [ -f "$PHASE2_CKPT" ]; then
    echo -e "${YELLOW}[SKIP]${NC} Phase 2 checkpoint exists"
else
    python scripts/train_with_timestamps.py \
        --model "$MODEL_NAME" \
        --checkpoint "$PHASE1_CKPT" \
        --timestamps "$TRAIN_TS" \
        --val-timestamps "$VAL_TS" \
        --output "$CHECKPOINT_DIR"
    
    echo -e "${GREEN}[OK]${NC} Phase 2 complete"
fi

# =============================================================================
# Copy Final Model
# =============================================================================

echo ""
echo "Copying final model to output directory..."

mkdir -p "$OUTPUT_DIR/models/$MODEL_NAME"

if [ -f "$PHASE2_CKPT" ]; then
    cp "$PHASE2_CKPT" "$OUTPUT_DIR/models/$MODEL_NAME/model.pt"
elif [ -f "$PHASE1_CKPT" ]; then
    cp "$PHASE1_CKPT" "$OUTPUT_DIR/models/$MODEL_NAME/model.pt"
fi

cp "$CONFIG_FILE" "$OUTPUT_DIR/models/$MODEL_NAME/config.yaml"

# =============================================================================
# Done!
# =============================================================================

echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo "Final model saved to:"
echo -e "  ${BLUE}$OUTPUT_DIR/models/$MODEL_NAME/model.pt${NC}"
echo ""
echo "To use the model:"
echo "  python scripts/inference.py \\"
echo "    --checkpoint $OUTPUT_DIR/models/$MODEL_NAME/model.pt \\"
echo "    --audio path/to/audio.wav"
echo ""
