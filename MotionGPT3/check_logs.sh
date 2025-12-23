#!/bin/bash
# Quick script to check the most recent MotionGPT SLURM logs

LOG_DIR="/scratch/network/is1893/mirror2_data/logs"

echo "=========================================="
echo "Most Recent Log Files"
echo "=========================================="
ls -lth "$LOG_DIR" | head -10
echo ""

# Get the most recent output and error files
LATEST_OUT=$(ls -t "$LOG_DIR"/*.out 2>/dev/null | head -1)
LATEST_ERR=$(ls -t "$LOG_DIR"/*.err 2>/dev/null | head -1)

if [ -n "$LATEST_OUT" ]; then
    echo "=========================================="
    echo "Most Recent Output Log: $(basename $LATEST_OUT)"
    echo "=========================================="
    tail -30 "$LATEST_OUT"
    echo ""
fi

if [ -n "$LATEST_ERR" ]; then
    echo "=========================================="
    echo "Most Recent Error Log: $(basename $LATEST_ERR)"
    echo "=========================================="
    tail -30 "$LATEST_ERR"
    echo ""
fi

echo "=========================================="
echo "Useful Commands:"
echo "=========================================="
echo "  # View last 50 lines of most recent output:"
echo "  tail -50 \$(ls -t $LOG_DIR/*.out | head -1)"
echo ""
echo "  # Follow most recent output in real-time:"
echo "  tail -f \$(ls -t $LOG_DIR/*.out | head -1)"
echo ""
echo "  # View most recent error log:"
echo "  tail -50 \$(ls -t $LOG_DIR/*.err | head -1)"
echo ""
echo "  # Search for errors in recent logs:"
echo "  grep -i error \$(ls -t $LOG_DIR/*.out | head -1)"
echo ""

