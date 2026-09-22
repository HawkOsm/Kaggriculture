#!/bin/bash
# Wait for the Kaggle daily quota to reset, then submit the best VALIDATED own agent.
# Usage: auto_submit.sh <file> "<message>"
cd /home/osm/Projects/Kaggriculture
FILE="${1:-submission/main_router_v2.py}"
MSG="${2:-adaptive-tape router v2: prefix-aligned family + measured shop routing (0.833 vs live 0.767 out-of-sample)}"
LOG=.claude/scratch/auto_submit.log
KG=.venv/bin/kaggle
echo "[$(date -u +%FT%TZ)] auto-submit armed for $FILE" >> "$LOG"
for i in $(seq 1 96); do            # up to ~16h of 10-min checks
  OUT=$(timeout 200 $KG competitions submit -c kaggriculture -f "$FILE" -m "$MSG" 2>&1)
  if echo "$OUT" | grep -q "Successfully submitted"; then
    echo "[$(date -u +%FT%TZ)] SUBMITTED $FILE" >> "$LOG"
    echo "$OUT" | grep -o "[0-9]* submissions remaining today" >> "$LOG"
    exit 0
  fi
  echo "[$(date -u +%FT%TZ)] attempt $i blocked (quota) — retrying in 10m" >> "$LOG"
  sleep 600
done
echo "[$(date -u +%FT%TZ)] gave up after 16h" >> "$LOG"
