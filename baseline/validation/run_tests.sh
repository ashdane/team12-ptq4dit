#!/bin/bash
# Team 12: PTQ4DiT Custom Triton Kernel Test Suite
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/src" ]; then
    ROOT_DIR="$SCRIPT_DIR"
    export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"
    TEST_DIR="$ROOT_DIR/tests"
elif [ -d "$SCRIPT_DIR/../src" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"
    TEST_DIR="$ROOT_DIR/tests"
else
    ROOT_DIR="$SCRIPT_DIR"
    export PYTHONPATH="$ROOT_DIR/sim/baseline_experiments:$PYTHONPATH"
    TEST_DIR="$ROOT_DIR/sim/baseline_experiments/tests"
fi

echo "================================================================================"
echo "Running PTQ4DiT Triton Kernel Unit Tests (32 tests)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Test Directory: $TEST_DIR"
echo "================================================================================"

python -m unittest discover -s "$TEST_DIR" -v
