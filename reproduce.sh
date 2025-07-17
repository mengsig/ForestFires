#!/bin/bash

if [ ! -x "$0" ]; then
    chmod +x "$0"
    bash "$0"
    exit 0
fi

if [ "$1" == "--IUseArchBtw" ]; then
    bash install.sh --IUseArchBtw
else
    bash install.sh
fi

source env.sh
echo "Running results for a single wind simulation..."
bash run.sh
echo "Finished running results for a single wind simulation!"

echo "Running results for mean behaviour without wind..."
bash fullrun.sh
echo "Finished running results for mean behaviour without wind!"

echo "Finished! Please find your results in src/results"
