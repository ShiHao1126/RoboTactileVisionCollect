#!/bin/bash

# if no arguments, use default values
if [ $# -eq 0 ]; then
    DEPTHS=(
    0.0004 0.0006 0.0008 0.0010 0.0012 0.0014 0.0016 0.0018
    )
else
    DEPTHS=("$@")
fi

PYTHON_SCRIPT="/home/ubuntu/workspace/flexiv/scripts/collect_DYS.py"

echo "=========================================="
echo "Batch collection with depths: ${DEPTHS[@]}"
echo "=========================================="

for depth in "${DEPTHS[@]}"; do
    echo ""
    echo "[$(date '+%H:%M:%S')] Running depth: ${depth}m"
    echo "----------------------------------------"
    
    python "$PYTHON_SCRIPT" --depth "$depth"
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✓ Completed: ${depth}m"
    else
        echo "[$(date '+%H:%M:%S')] ✗ Failed: ${depth}m"
        exit 1
    fi

    sleep 10
done



echo ""
echo "=========================================="
echo "[$(date '+%H:%M:%S')] All done!"
echo "=========================================="