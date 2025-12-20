#!/bin/bash

# Real-time loss tracking - continuously extracts and displays Secret loss trend
# This is the KEY METRIC to watch for training improvement

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         STEGASTAMP TRAINING - SECRET LOSS TRACKER           ║"
echo "║  (This is the key indicator of whether training is working) ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Watching: /tmp/train_output.log"
echo "Metric: Secret loss (target: DECREASING over time)"
echo "Last updated: $(date)"
echo ""
echo "Step | Secret Loss | Trend | Phase"
echo "-----|-------------|-------|-------"

tail -f /tmp/train_output.log | grep -E "\[P[1-3]\].*Step" | while read line; do
    # Extract values
    STEP=$(echo "$line" | grep -oP 'Step\s+\K[0-9]+')
    SECRET=$(echo "$line" | grep -oP 'Secret:\K[0-9.]+')
    PHASE=$(echo "$line" | grep -oP '\[\K[P0-9]+' | tr '\n' ' ')

    if [ -n "$STEP" ] && [ -n "$SECRET" ]; then
        # Determine trend with arrows
        if (( $(echo "$SECRET < 0.65" | bc -l) )); then
            TREND="↓ Good"
        elif (( $(echo "$SECRET < 0.69" | bc -l) )); then
            TREND="→ Stable"
        else
            TREND="↑ Problem"
        fi

        printf "%5d | %11.6f | %6s | %s\n" "$STEP" "$SECRET" "$TREND" "$PHASE"
    fi
done
