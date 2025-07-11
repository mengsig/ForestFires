#!/bin/bash
set -euo pipefail
source env.sh

num_cores=$(nproc)
if (( num_cores > 1 )); then
  max_jobs=$(( num_cores - 1 ))
  max_jobs=10
else
  max_jobs=1
fi
echo "▶ Allowing up to $max_jobs parallel jobs (cores minus one)."

#— Helper: after launching a job in background, wait until we drop below max_jobs
throttle() {
  # Loop until the number of running background jobs is < max_jobs
  while (( $(jobs -rp | wc -l) >= max_jobs )); do
    sleep 0.5
  done
}

XLEN=250
YLEN=250
SAVENAME="250_4neighbors_full"
#source env.sh
python src/scripts/create_adjacency.py "${XLEN}x${YLEN}" "$SAVENAME"

CENTRALITIES=(domirank random degree bonacich)
echo "Computing Fuel-Breaks for all centralities..."
for c in "${CENTRALITIES[@]}"; do
  (
    echo "=== Computing Fuel-Breaks for $c ==="
    python src/scripts/generate_fuel_breaks.py "${XLEN}x${YLEN}" "${SAVENAME}" "$c"
  )     # <–– running serially 
done

echo "✅ Done Fuel-Breaks!"

#— Example: parallel simulations (throttled)
PERC=(0 5 10 15 20 25 30)
echo "▶ Simulating fire-spread for all centralities and fractions…"
for perc in "${PERC[@]}"; do
  for c in "${CENTRALITIES[@]}"; do
    (
      frac=$(awk "BEGIN { printf \"%.2f\", ${perc} }")
      echo "=== Simulating $c @ fuel_break_fraction=$frac ==="
      python src/scripts/simulate.py \
        "${XLEN}x${YLEN}" \
        "$SAVENAME" \
        "$c" \
        "$perc"
    ) &
    throttle
  done
done

# wait for all simulation jobs
wait
echo "✅ All simulations complete!"

echo "Generating plots..."

python src/scripts/generate_plots.py \
        "${XLEN}x${YLEN}" \
        "$SAVENAME" \

echo "✅ All simulations complete!"
