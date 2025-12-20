#!/bin/bash
# Background monitoring - shows training progress every 10 minutes

echo "StegaStamp Training Monitor (updates every 10 minutes)"
echo "Starting at $(date)"
echo ""

prev_step=0
prev_loss=0

while true; do
  sleep 600  # 10 minutes

  if [ -f /tmp/train_output.log ]; then
    latest=$(tail -1 /tmp/train_output.log)

    if [[ $latest =~ Step\ +([0-9]+) ]]; then
      step="${BASH_REMATCH[1]}"

      if [[ $latest =~ Secret:([0-9.]+) ]]; then
        loss="${BASH_REMATCH[1]}"
        elapsed=$((step * 12 / 1000))  # approx ms per step
        eta_hours=$((($elapsed * (140000 - step)) / 360000000))

        printf "[$(date +%H:%M:%S)] Step %d/140000 | Secret Loss: %.4f | ETA: ~%dh | Progress: %.1f%%\n" \
          "$step" "$loss" "$eta_hours" "$(echo "scale=1; $step * 100 / 140000" | bc)"

        if [ $step -ge 1000 ] && [ $((step % 5000)) -lt 50 ]; then
          echo "CHECKPOINT MILESTONE: Step $step - Saving models"
        fi
      fi
    fi
  fi
done
