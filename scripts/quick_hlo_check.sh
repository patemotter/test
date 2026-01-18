#!/bin/bash
# Quick HLO inspection for fp8 2D quantized matmul kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "FP8 2D Quantized Matmul - HLO Quick Check"
echo "=========================================="
echo ""

# Check if compilation output exists
if [ -d "$PROJECT_DIR/compilation_output" ] && [ -n "$(ls -A $PROJECT_DIR/compilation_output/*.txt 2>/dev/null)" ]; then
    echo "✓ Found existing HLO dumps in compilation_output/"
    echo ""
    echo "Analyzing existing dumps..."
    echo ""
    python3 "$SCRIPT_DIR/analyze_hlo.py" "$PROJECT_DIR"/compilation_output/*_hlo.txt
else
    echo "No existing HLO dumps found."
    echo ""
    echo "Attempting to generate HLO dumps..."
    echo "(This requires JAX to be installed)"
    echo ""

    # Try to run compilation inspector
    if python3 -c "import jax" 2>/dev/null; then
        cd "$PROJECT_DIR"
        python3 "$SCRIPT_DIR/inspect_fp8_2d_kernel_compilation.py"

        echo ""
        echo "HLO dumps generated! Analyzing..."
        echo ""
        python3 "$SCRIPT_DIR/analyze_hlo.py" compilation_output/*_hlo.txt
    else
        echo "✗ JAX not installed. Cannot generate HLO dumps."
        echo ""
        echo "To generate HLO dumps:"
        echo "  1. Install JAX: pip install jax jaxlib"
        echo "  2. Run: python scripts/inspect_fp8_2d_kernel_compilation.py"
        echo "  3. Run: python scripts/analyze_hlo.py compilation_output/*.txt"
        echo ""
        echo "Or run this script on a machine with JAX installed."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Quick Check Complete!"
echo "=========================================="
echo ""
echo "📝 See scripts/README_HLO_INSPECTION.md for detailed analysis guide"
echo ""
