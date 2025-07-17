#!/bin/bash
set -euo pipefail

XLEN=250
YLEN=250
FUEL_BREAK_FRACTION=15
SAVENAME="new_250_8neighbors_${FUEL_BREAK_FRACTION}"
source env.sh
python src/scripts/create_adjacency.py "${XLEN}x${YLEN}" "$SAVENAME"

CENTRALITIES=(protected_domirank)
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

