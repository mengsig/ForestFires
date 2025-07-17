#!/bin/bash
set -euo pipefail
source env.sh

# USER PARAMETERS
XLEN=250
YLEN=250
SAVENAME="250_8neighbors"
CENTRALITIES=(protected_domirank domirank random degree bonacich)
PERC=(0 15)

#activating user environment
source env.sh

num_cores=$(nproc)
if (( num_cores > 1 )); then
  max_jobs=$(( num_cores - 1 ))
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

#source env.sh
python src/scripts/create_adjacency.py "${XLEN}x${YLEN}" "$SAVENAME"

echo "Computing Fuel-Breaks for all centralities..."
for c in "${CENTRALITIES[@]}"; do
  (
    echo "=== Computing Fuel-Breaks for $c ==="
    python src/scripts/generate_fuel_breaks.py "${XLEN}x${YLEN}" "${SAVENAME}" "$c"
  )     # <–– running serially 
done

echo "✅ Done Creating Fuel-Breaks!"


echo "▶ Simulating fire-spread for all centralities and fractions…"
for perc in "${PERC[@]}"; do
  for c in "${CENTRALITIES[@]}"; do
    # if perc==0, only run domirank
    if [[ $perc -eq 0 && "$c" != "domirank" ]]; then
      continue
    fi

    (
      # format into 0.00 or 15.00 etc.
      frac=$(awk "BEGIN { printf \"%.2f\", $perc }")
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

#echo "Generating plots..."
#
#python src/scripts/generate_plots.py \
#        "${XLEN}x${YLEN}" \
#        "$SAVENAME" \
#
echo "✅ All simulations complete!"
