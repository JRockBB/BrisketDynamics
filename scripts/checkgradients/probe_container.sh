#!/usr/bin/env bash
# Probe whether the sweep container is progressing (CPU ticks advancing on
# the julia/python worker) or stuck. Two samples, 45 s apart.
C=${1:-serene_gagarin}
sample() {
  docker exec "$C" sh -c '
    for p in /proc/[0-9]*; do
      st=$(cat $p/stat 2>/dev/null) || continue
      set -- $st
      echo "$(( $14 + $15 )) $(cat $p/comm 2>/dev/null)"
    done | sort -rn | head -3'
}
echo "sample 1:"; sample
sleep 45
echo "sample 2:"; sample
