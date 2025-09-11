#!/bin/bash
# Wrapper script to run the diffusion pipeline with expert demonstrations
# This script ensures the correct conda environments are activated

set -e  # Exit on error

echo "🚀 Starting Expert Diffusion Pipeline"
echo "======================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install conda first."
    exit 1
fi

# Function to check if environment exists
check_env() {
    local env_name=$1
    if ! conda env list | grep -q "^${env_name} "; then
        echo "❌ Error: Conda environment '${env_name}' not found."
        echo "Please create the environment first."
        exit 1
    fi
}

# Check required environments
echo "🔍 Checking conda environments..."
check_env "prbenchIL"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Parse arguments to check if we need expert data
NEEDS_EXPERT=false
for arg in "$@"; do
    if [[ "$arg" == "--data-type=expert" ]] || [[ "$arg" == "expert" ]]; then
        NEEDS_EXPERT=true
        break
    fi
done

# Check if previous argument was --data-type
PREV_ARG=""
for arg in "$@"; do
    if [[ "$PREV_ARG" == "--data-type" ]] && [[ "$arg" == "expert" ]]; then
        NEEDS_EXPERT=true
        break
    fi
    PREV_ARG="$arg"
done

echo "📋 Running pipeline with prbenchIL environment (supports both expert and random data)..."
# Use prbenchIL environment for everything since it now has working bilevel planning imports
conda run -n prbenchIL python "${SCRIPT_DIR}/run_diffusion_pipeline.py" "$@"
echo "✅ Pipeline completed successfully!"

echo "======================================"
echo "🎉 All done!"
