#!/bin/bash
# SBM Toolkit - Phase Diagram Generation Script
#
# Usage: ./generate_phase_diagram.sh [--data-path DIR] [--output-dir DIR]
#

# Default values
DATA_DIR="data/"
OUTPUT_DIR="results/"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-path)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "SBM Toolkit - Phase Diagram Generation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-path PATH    Path to data directory (default: data/)"
            echo "  --output-dir DIR    Output directory (default: results/)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo "SBM Phase Diagram Generation"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run Python script
python /home/ansatz/data/code/2024-SBM/sbm_toolkit/scripts/generate_phase_diagram.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"
