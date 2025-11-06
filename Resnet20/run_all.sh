#!/bin/bash


PERCENTS=(0.1 0.2 0.3 0.4 0.5)


DIFFICULTIES=(e h)

LOGFILE="logs.txt"

for DIFF in "${DIFFICULTIES[@]}"; do
  for P in "${PERCENTS[@]}"; do
    
    P_INT=$(awk "BEGIN {print int($P*100)}")

    OUTFILE="${DIFF}${P_INT}.txt"

    echo "Running with --difficulty $DIFF --percent $P" | tee -a "$LOGFILE"
    echo "Saving output to $OUTFILE" | tee -a "$LOGFILE"

    python3 pruning.py --difficulty "$DIFF" --percent "$P" | tee "$OUTFILE"
  done
done

OUTFILE="a100.txt"
DIFF="e"
P="1.0"
echo "Running with --difficulty $DIFF --percent $P" | tee -a "$LOGFILE"
python3 pruning.py --difficulty "$DIFF" --percent "$P" | tee "$OUTFILE"

echo "DONE" | tee -a "$LOGFILE"