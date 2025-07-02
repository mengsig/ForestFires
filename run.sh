#!/bin/bash
set -euo pipefail

XLEN=300
YLEN=300
SAVENAME="300_4neighbors"
source env.sh
python src/scripts/create_adjacency.py "${XLEN}x${YLEN}" "$SAVENAME"

CENTRALITIES=(domirank random degree bonacich)
echo "Computing Fuel-Breaks for all centralities..."
for c in "${CENTRALITIES[@]}"; do
  (
    echo "=== Computing Fuel-Breaks for $c ==="
    python src/scripts/generate_fuel_breaks.py "${XLEN}x${YLEN}" "${SAVENAME}" "$c"
  ) &   # <–– running subshell in background
done

wait    # <–– hold until all background jobs finish

echo "Computed Fuel-Breaks for all centralities!"

echo "Simulating fire-spreading for all centralities..."
for c in "${CENTRALITIES[@]}"; do
  (
    echo "=== Simulating fire-spreading for $c ==="
    python src/scripts/simulate.py "${XLEN}x${YLEN}" "${SAVENAME}" "$c"
  ) &   # <–– running subshell in background
done

wait    # <–– hold until all background jobs finish

echo "Finished simulating fire-spreading for all centralities!"

echo "Done!"

