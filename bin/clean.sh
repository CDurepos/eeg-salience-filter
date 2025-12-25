#!/bin/bash
# Clean generated files (preserves preprocessed data and benchmark)
#
# Removes:
#   - All filtered data
#   - All teacher/student models (keeps benchmark)
#   - Outputs and plots
#
# Use clean_all.sh to remove everything including benchmark and preprocessed data

set -e

echo "Cleaning generated files..."

# Remove filtered data for all teachers
rm -rf data/filtered_eegnet data/filtered_resnet data/filtered_tst

# Remove teacher/student models (keep benchmark)
rm -rf models/eegnet models/resnet models/tst

# Remove outputs (except evaluation results - regenerated on next run)
rm -rf outputs/eegnet outputs/resnet outputs/tst

# Remove plots
rm -rf plots

echo "✓ Cleaned. Preserved: data/unfiltered/, models/benchmark.pt"
echo ""
echo "To re-run the study:"
echo "  ./bin/run.sh"
echo ""
echo "To clean EVERYTHING (including benchmark and preprocessed data):"
echo "  ./bin/clean_all.sh"
