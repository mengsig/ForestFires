#!/bin/bash
set -euo pipefail

XLEN=100
YLEN=100
SAVENAME="100_4neighbors"
FUEL_BREAK_FRACTION=15
source env.sh
python src/scripts/create_adjacency.py "${XLEN}x${YLEN}" "$SAVENAME"

CENTRALITIES=(domirank random degree bonacich)
echo "Computing Fuel-Breaks for all centralities..."
for c in "${CENTRALITIES[@]}"; do
  (
    echo "=== Computing Fuel-Breaks for $c ==="
    python src/scripts/generate_fuel_breaks.py "${XLEN}x${YLEN}" "${SAVENAME}" "$c"
  )     # <–– running serially 
done

echo "Computed Fuel-Breaks for all centralities!"

echo "Simulating fire-spreading for all centralities..."
for c in "${CENTRALITIES[@]}"; do
  (
    echo "=== Simulating fire-spreading for $c ==="
    python src/scripts/simulate.py "${XLEN}x${YLEN}" "${SAVENAME}" "$c" "$FUEL_BREAK_FRACTION"
  ) &   # <–– running subshell in background
done

wait    # <–– hold until all background jobs finish

echo "Finished simulating fire-spreading for all centralities!"

echo "Done!"

