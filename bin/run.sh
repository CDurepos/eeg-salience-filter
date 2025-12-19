#!/bin/bash
# Complete pipeline script for EEG Salience Filter
# Runs all scripts in the correct order

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    echo "Please run bin/setup.sh first to download the dataset."
    exit 1
fi

# Check if arl-eegmodels exists
if [ ! -d "src/arl-eegmodels" ]; then
    print_error "ARL EEGModels not found in src/arl-eegmodels"
    echo "Please run bin/setup.sh first to clone the repository."
    exit 1
fi

# Get DATA_PATH from .env
DATA_PATH=$(grep DATA_PATH .env | cut -d '=' -f2)
if [ -z "$DATA_PATH" ]; then
    print_error "DATA_PATH not found in .env file"
    exit 1
fi

print_step "EEG Salience Filter Pipeline"
echo "Dataset path: $DATA_PATH"
echo ""

# Step 1: Preprocess
print_step "Step 1/6: Preprocessing Data"
if [ -d "data/unfiltered" ] && [ -f "data/unfiltered/epochs.npy" ]; then
    print_warning "Preprocessed data already exists. Skipping preprocessing."
    echo "  To re-run preprocessing, delete data/unfiltered/ first."
else
    python src/preprocess.py --data-path "$DATA_PATH/adhdata.csv" --out-dir data
    if [ $? -eq 0 ]; then
        print_success "Preprocessing completed"
    else
        print_error "Preprocessing failed"
        exit 1
    fi
fi
echo ""

# Step 2: Train Teacher
print_step "Step 2/6: Training Teacher Model"
if [ -f "models/teacher/model.h5" ]; then
    print_warning "Teacher model already exists. Skipping training."
    echo "  To re-train, delete models/teacher/ first."
else
    python src/train/teacher.py
    if [ $? -eq 0 ]; then
        print_success "Teacher model training completed"
    else
        print_error "Teacher model training failed"
        exit 1
    fi
fi
echo ""

# Step 3: Process (Generate Salience Maps & Filter)
print_step "Step 3/6: Processing (Salience Maps & Filtering)"
if [ -d "data/filtered" ] && [ -f "data/filtered/epochs.npy" ]; then
    print_warning "Filtered data already exists. Skipping processing."
    echo "  To re-run processing, delete data/filtered/ first."
else
    python src/process.py
    if [ $? -eq 0 ]; then
        print_success "Processing completed"
    else
        print_error "Processing failed"
        exit 1
    fi
fi
echo ""

# Step 4: Train Student
print_step "Step 4/6: Training Student Model"
if [ -f "models/student/model.h5" ]; then
    print_warning "Student model already exists. Skipping training."
    echo "  To re-train, delete models/student/ first."
else
    python src/train/student.py
    if [ $? -eq 0 ]; then
        print_success "Student model training completed"
    else
        print_error "Student model training failed"
        exit 1
    fi
fi
echo ""

# Step 5: Train Benchmark
print_step "Step 5/6: Training Benchmark Model"
if [ -f "models/benchmark/model.h5" ]; then
    print_warning "Benchmark model already exists. Skipping training."
    echo "  To re-train, delete models/benchmark/ first."
else
    python src/train/benchmark.py
    if [ $? -eq 0 ]; then
        print_success "Benchmark model training completed"
    else
        print_error "Benchmark model training failed"
        exit 1
    fi
fi
echo ""

# Step 6: Evaluate
print_step "Step 6/6: Evaluating Models"
python src/evaluate.py
if [ $? -eq 0 ]; then
    print_success "Evaluation completed"
else
    print_error "Evaluation failed"
    exit 1
fi
echo ""

# Final summary
print_step "Pipeline Complete!"
echo ""
echo "Results saved to:"
echo "  - Models: models/teacher/, models/student/, models/benchmark/"
echo "  - Data: data/unfiltered/, data/filtered/"
echo "  - Evaluation: outputs/evaluation_comparison.json"
echo ""
print_success "All steps completed successfully!"

