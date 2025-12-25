#!/bin/bash
# Complete pipeline for EEG Salience Filter Study (PyTorch)
#
# Structure:
#   - 1 Benchmark EEGNet (baseline)
#   - 3 Teachers: EEGNet, ResNet, TST (from braindecode/tsai)
#   - 3 Students: One per teacher (all EEGNet on filtered data)
#
# Uses:
#   - braindecode for EEGNet
#   - tsai for ResNet and TST
#   - captum for salience maps (Integrated Gradients)
#
# Usage:
#   ./bin/run.sh           # Run full pipeline
#   ./bin/run.sh --force   # Force re-run all steps

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

step() { echo -e "\n${BLUE}=== $1 ===${NC}"; }
ok() { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
err() { echo -e "${RED}✗ $1${NC}"; exit 1; }

# Check environment
[ ! -f ".env" ] && err ".env not found! Run bin/setup.sh first."
[ ! -d "src/arl-eegmodels" ] && err "arl-eegmodels not found! Run bin/setup.sh first."

DATA_PATH=$(grep DATA_PATH .env | cut -d '=' -f2)
[ -z "$DATA_PATH" ] && err "DATA_PATH not set in .env"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  EEG Salience Filter - Multi-Teacher Study (PyTorch)      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# ============================================================================
# Step 1: Preprocessing
# ============================================================================
step "Step 1: Preprocessing"
if [ -f "data/unfiltered/epochs.npy" ] && [ "$1" != "--force" ]; then
    warn "Preprocessed data exists. Skipping. (use --force to redo)"
else
    python src/preprocess.py --data-path "$DATA_PATH/adhdata.csv" --out-dir data
    ok "Preprocessing complete"
fi

# ============================================================================
# Step 2: Train Benchmark
# ============================================================================
step "Step 2: Train Benchmark EEGNet (braindecode)"
if [ -f "models/benchmark.pt" ] && [ "$1" != "--force" ]; then
    warn "Benchmark exists. Skipping."
else
    python src/train/benchmark.py
    ok "Benchmark trained"
fi

# ============================================================================
# Step 3-5: Train Teachers
# ============================================================================
for MODEL in eegnet resnet tst; do
    step "Training $MODEL Teacher"
    if [ -f "models/$MODEL/teacher.pt" ] && [ "$1" != "--force" ]; then
        warn "$MODEL teacher exists. Skipping."
    else
        python src/train/teacher.py --model $MODEL
        ok "$MODEL teacher trained"
    fi
done

# ============================================================================
# Step 6-8: Process (generate salience maps & filtered data)
# ============================================================================
for MODEL in eegnet resnet tst; do
    step "Processing: $MODEL → Filtered Data (Captum)"
    if [ -f "data/filtered_$MODEL/epochs.npy" ] && [ "$1" != "--force" ]; then
        warn "Filtered data for $MODEL exists. Skipping."
    else
        python src/process.py --teacher $MODEL
        ok "$MODEL filtering complete"
    fi
done

# ============================================================================
# Step 9-11: Train Students
# ============================================================================
for MODEL in eegnet resnet tst; do
    step "Training Student (from $MODEL teacher)"
    if [ -f "models/$MODEL/student.pt" ] && [ "$1" != "--force" ]; then
        warn "Student for $MODEL exists. Skipping."
    else
        python src/train/student.py --teacher $MODEL
        ok "Student ($MODEL) trained"
    fi
done

# ============================================================================
# Step 12: Evaluate All Models
# ============================================================================
step "Evaluating All Models"
python src/evaluate.py
ok "Evaluation complete"

# ============================================================================
# Step 13: Generate Summary
# ============================================================================
step "Generating Summary"
python src/summary.py
ok "Summary generated"

# ============================================================================
# Done
# ============================================================================
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Pipeline Complete!                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results:"
echo "  - Models:      models/benchmark.pt, models/{eegnet,resnet,tst}/"
echo "  - Data:        data/unfiltered/, data/filtered_{eegnet,resnet,tst}/"
echo "  - Outputs:     outputs/evaluation_results.json"
echo "  - Plots:       plots/"
echo ""
