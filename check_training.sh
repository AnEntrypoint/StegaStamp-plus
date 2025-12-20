#!/bin/bash
echo "============================================================"
echo "TRAINING STATUS CHECK"
echo "============================================================"
echo ""

# Check if training is running
if pgrep -f "python3 train_100bit.py" > /dev/null; then
    PID=$(pgrep -f "python3 train_100bit.py")
    CPU=$(ps aux | grep $PID | grep -v grep | awk '{print $3}')
    MEM=$(ps aux | grep $PID | grep -v grep | awk '{print $4}')
    echo "✓ Training RUNNING"
    echo "  PID: $PID | CPU: ${CPU}% | Memory: ${MEM}%"
    echo ""
else
    echo "✗ Training NOT RUNNING"
    echo ""
fi

# Show latest training output
echo "Latest training logs (last 10 lines):"
echo "---"
tail -10 /tmp/train_output.log 2>/dev/null | grep -E "\[|Step|CHECKPOINT" || echo "  (waiting for output...)"
echo ""

# List checkpoints created so far
echo "Checkpoints created:"
ls -lh /home/user/StegaStamp-plus/encoder_100bit_step_*.keras 2>/dev/null | awk '{print "  Step " substr($9, match($9, /[0-9]+\.keras/)) " - " $5}' | sort -V || echo "  (none yet)"
echo ""

# Estimate progress
LATEST_STEP=$(ls -1 /home/user/StegaStamp-plus/encoder_100bit_step_*.keras 2>/dev/null | sed 's/.*step_//' | sed 's/.keras//' | sort -n | tail -1)
if [ -n "$LATEST_STEP" ]; then
    PROGRESS=$((LATEST_STEP * 100 / 140000))
    REMAINING=$((140000 - LATEST_STEP))
    echo "Progress: $LATEST_STEP / 140000 steps ($PROGRESS%)"
    echo "Remaining: $REMAINING steps"
else
    echo "Progress: Initializing..."
fi
echo ""
echo "============================================================"
