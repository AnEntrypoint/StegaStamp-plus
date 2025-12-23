#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "Usage: ./train.sh [100|256]"
    echo "  ./train.sh 100  - Train 100-bit baseline"
    echo "  ./train.sh 256  - Train 256-bit full model"
    exit 1
fi

BITS=$1

if [ "$BITS" = "100" ]; then
    echo "Starting 100-bit training..."
    python3 train_100bit.py
elif [ "$BITS" = "256" ]; then
    echo "Starting 256-bit training..."
    python3 train_256bit.py
else
    echo "Invalid choice. Use 100 or 256"
    exit 1
fi
