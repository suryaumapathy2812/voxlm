#!/bin/bash
# =============================================================================
# VoxLM Quick Training Script
# =============================================================================
# One-command training: downloads data, generates timestamps, trains model.
#
# Usage:
#   ./scripts/quick_train.sh                          # Uses default config (voxlm-2b)
#   ./scripts/quick_train.sh configs/voxlm-2b.yaml    # Specify config file
#   ./scripts/quick_train.sh configs/voxlm-9b-global.yaml  # Production model
#
# With uv (recommended):
#   uv sync                                           # Install dependencies first
#   UV_RUN="uv run" ./scripts/quick_train.sh          # Use uv for Python
#
# What this script does:
#   1. Validates environment (Python, CUDA, dependencies)
#   2. Downloads LibriSpeech data (if not present)
#   3. Generates word-level timestamps using Whisper
#   4. Phase 1: Trains basic transcription
#   5. Phase 2: Trains timestamp alignment
#   6. Copies final model to output directory
#
# Requirements:
#   - Python 3.10+
#   - CUDA (optional, but recommended)
#   - ~10GB disk space for data
#   - ~16GB VRAM for voxlm-2b, ~24GB for voxlm-9b
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Configuration
# =============================================================================

# Default config file
CONFIG_FILE="${1:-configs/voxlm-2b.yaml}"

# Script directory (for relative paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Python command (use UV_RUN env var if set, otherwise python3)
# Usage: UV_RUN="uv run" ./scripts/quick_train.sh
PYTHON="${UV_RUN:-python3}"

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  VoxLM Quick Training${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo -e "Config: ${GREEN}$CONFIG_FILE${NC}"
echo -e "Project: ${GREEN}$PROJECT_DIR${NC}"
echo ""

# =============================================================================
# Helper Functions
# =============================================================================

log_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse YAML value using Python (handles nested YAML properly)
parse_yaml() {
    local file=$1
    local key=$2
    local default=$3
    
    # Use Python to parse YAML properly
    local value=$($PYTHON -c "
import yaml
import sys

def get_nested(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d

with open('$file') as f:
    config = yaml.safe_load(f)

# Handle nested keys like 'training.phase1.epochs'
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

# =============================================================================
# Step 0: Validate Environment
# =============================================================================

log_step "Validating environment..."

# Check Python
# Check if using uv or direct python
if [ "$UV_RUN" = "uv run" ]; then
    log_success "Using uv for Python"
    PYTHON_VERSION=$($PYTHON python --version 2>&1 | cut -d' ' -f2)
else
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi
log_success "Python $PYTHON_VERSION"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -la configs/*.yaml 2>/dev/null || echo "  No configs found in configs/"
    exit 1
fi
log_success "Config file: $CONFIG_FILE"

# Check CUDA
if $PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    log_success "CUDA available: $GPU_NAME"
    DEVICE="cuda"
else
    log_warning "CUDA not available, using CPU (training will be slow)"
    DEVICE="cpu"
fi

# Check dependencies
log_step "Checking dependencies..."
$PYTHON -c "import torch" 2>/dev/null || { log_error "PyTorch not installed"; exit 1; }
$PYTHON -c "import transformers" 2>/dev/null || { log_error "transformers not installed"; exit 1; }
$PYTHON -c "import yaml" 2>/dev/null || { log_error "PyYAML not installed. Run: pip install pyyaml"; exit 1; }
log_success "All dependencies available"

# =============================================================================
# Step 1: Parse Configuration
# =============================================================================

log_step "Parsing configuration..."

# Parse key values from config (using dot notation for nested keys)
MODEL_NAME=$(parse_yaml "$CONFIG_FILE" "model.name" "voxlm-2b")
DATA_DIR=$(parse_yaml "$CONFIG_FILE" "data.data_dir" "./data")
TRAIN_SPLIT=$(parse_yaml "$CONFIG_FILE" "data.train_split" "train-clean-100")
VAL_SPLIT=$(parse_yaml "$CONFIG_FILE" "data.val_split" "dev-clean")
TIMESTAMPS_DIR=$(parse_yaml "$CONFIG_FILE" "data.timestamps_dir" "./data/timestamps")
CHECKPOINT_DIR=$(parse_yaml "$CONFIG_FILE" "output.checkpoint_dir" "./checkpoints")
FINAL_MODEL_DIR=$(parse_yaml "$CONFIG_FILE" "output.final_model_dir" "./output/models")
WHISPER_MODEL=$(parse_yaml "$CONFIG_FILE" "data.whisper_model" "small")

# Phase 1 settings (nested under training.phase1)
P1_EPOCHS=$(parse_yaml "$CONFIG_FILE" "training.phase1.epochs" "10")
P1_BATCH=$(parse_yaml "$CONFIG_FILE" "training.phase1.batch_size" "8")
P1_LR=$(parse_yaml "$CONFIG_FILE" "training.phase1.learning_rate" "1e-4")

# Phase 2 settings (nested under training.phase2)
P2_EPOCHS=$(parse_yaml "$CONFIG_FILE" "training.phase2.epochs" "5")
P2_BATCH=$(parse_yaml "$CONFIG_FILE" "training.phase2.batch_size" "16")
P2_LR=$(parse_yaml "$CONFIG_FILE" "training.phase2.learning_rate" "5e-5")

echo "  Model: $MODEL_NAME"
echo "  Data: $DATA_DIR"
echo "  Train split: $TRAIN_SPLIT"
echo "  Val split: $VAL_SPLIT"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Final output: $FINAL_MODEL_DIR"

# =============================================================================
# Step 2: Download Data (if needed)
# =============================================================================

log_step "Checking data..."

TRAIN_DATA_PATH="$DATA_DIR/LibriSpeech/$TRAIN_SPLIT"
VAL_DATA_PATH="$DATA_DIR/LibriSpeech/$VAL_SPLIT"

if [ -d "$TRAIN_DATA_PATH" ] && [ -d "$VAL_DATA_PATH" ]; then
    log_success "Data already exists"
else
    log_warning "Data not found, downloading..."
    
    # Create data directory
    mkdir -p "$DATA_DIR"
    
    # Download using the download script
    if [ -f "scripts/download_data.py" ]; then
        $PYTHON scripts/download_data.py --output "$DATA_DIR" --splits "$TRAIN_SPLIT" "$VAL_SPLIT"
    else
        # Fallback: direct download
        echo "Downloading LibriSpeech $TRAIN_SPLIT..."
        cd "$DATA_DIR"
        
        if [ "$TRAIN_SPLIT" = "train-clean-100" ]; then
            wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz
            tar -xzf train-clean-100.tar.gz
        fi
        
        if [ "$VAL_SPLIT" = "dev-clean" ]; then
            wget -c https://www.openslr.org/resources/12/dev-clean.tar.gz
            tar -xzf dev-clean.tar.gz
        fi
        
        cd "$PROJECT_DIR"
    fi
    
    log_success "Data downloaded"
fi

# =============================================================================
# Step 3: Generate Timestamps (if needed)
# =============================================================================

log_step "Checking timestamps..."

TRAIN_TIMESTAMPS="$TIMESTAMPS_DIR/$TRAIN_SPLIT/timestamps_manifest.json"
VAL_TIMESTAMPS="$TIMESTAMPS_DIR/$VAL_SPLIT/timestamps_manifest.json"

if [ -f "$TRAIN_TIMESTAMPS" ] && [ -f "$VAL_TIMESTAMPS" ]; then
    log_success "Timestamps already exist"
else
    log_warning "Generating timestamps with Whisper..."
    
    # Generate for training split (uses --config with --split override)
    if [ ! -f "$TRAIN_TIMESTAMPS" ]; then
        echo "Generating timestamps for $TRAIN_SPLIT..."
        $PYTHON scripts/generate_timestamps.py \
            --config "$CONFIG_FILE" \
            --split "$TRAIN_SPLIT" \
            --device "$DEVICE" \
            --fast
    fi
    
    # Generate for validation split (uses --config with --split override)
    if [ ! -f "$VAL_TIMESTAMPS" ]; then
        echo "Generating timestamps for $VAL_SPLIT..."
        $PYTHON scripts/generate_timestamps.py \
            --config "$CONFIG_FILE" \
            --split "$VAL_SPLIT" \
            --device "$DEVICE" \
            --fast
    fi
    
    log_success "Timestamps generated"
fi

# =============================================================================
# Step 4: Phase 1 Training (Basic Transcription)
# =============================================================================

log_step "Phase 1: Training basic transcription..."

PHASE1_CHECKPOINT="$CHECKPOINT_DIR/$MODEL_NAME/best.pt"

if [ -f "$PHASE1_CHECKPOINT" ]; then
    log_warning "Phase 1 checkpoint exists, skipping (delete to retrain)"
else
    echo "Training $MODEL_NAME for $P1_EPOCHS epochs..."
    
    # Use --config directly, with device override for runtime detection
    $PYTHON scripts/train.py \
        --config "$CONFIG_FILE" \
        --device "$DEVICE"
    
    log_success "Phase 1 complete"
fi

# =============================================================================
# Step 5: Phase 2 Training (Timestamp Alignment)
# =============================================================================

log_step "Phase 2: Training timestamp alignment..."

PHASE2_CHECKPOINT="$CHECKPOINT_DIR/$MODEL_NAME/best_phase2.pt"

if [ -f "$PHASE2_CHECKPOINT" ]; then
    log_warning "Phase 2 checkpoint exists, skipping (delete to retrain)"
else
    echo "Training alignment for $P2_EPOCHS epochs..."
    
    # Use --config directly, with checkpoint and device overrides
    $PYTHON scripts/train_with_timestamps.py \
        --config "$CONFIG_FILE" \
        --checkpoint "$PHASE1_CHECKPOINT" \
        --device "$DEVICE"
    
    log_success "Phase 2 complete"
fi

# =============================================================================
# Step 6: Copy Final Model
# =============================================================================

log_step "Copying final model to output directory..."

# Create output directory
mkdir -p "$FINAL_MODEL_DIR/$MODEL_NAME"

# Copy the best checkpoint
if [ -f "$PHASE2_CHECKPOINT" ]; then
    cp "$PHASE2_CHECKPOINT" "$FINAL_MODEL_DIR/$MODEL_NAME/model.pt"
    log_success "Final model: $FINAL_MODEL_DIR/$MODEL_NAME/model.pt"
elif [ -f "$PHASE1_CHECKPOINT" ]; then
    cp "$PHASE1_CHECKPOINT" "$FINAL_MODEL_DIR/$MODEL_NAME/model.pt"
    log_warning "Using Phase 1 model (Phase 2 not completed)"
    log_success "Final model: $FINAL_MODEL_DIR/$MODEL_NAME/model.pt"
else
    log_error "No checkpoint found!"
    exit 1
fi

# Copy config for reference
cp "$CONFIG_FILE" "$FINAL_MODEL_DIR/$MODEL_NAME/config.yaml"

# =============================================================================
# Done!
# =============================================================================

echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""
echo "Final model saved to:"
echo -e "  ${BLUE}$FINAL_MODEL_DIR/$MODEL_NAME/model.pt${NC}"
echo ""
echo "To run inference:"
echo -e "  ${YELLOW}python scripts/inference.py \\${NC}"
echo -e "  ${YELLOW}  --checkpoint $FINAL_MODEL_DIR/$MODEL_NAME/model.pt \\${NC}"
echo -e "  ${YELLOW}  --audio path/to/audio.wav${NC}"
echo ""
