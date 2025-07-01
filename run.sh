#!/bin/bash

XLEN=500
YLEN=500
SAVENAME="500Test"
source env.sh
python src/scripts/create_adjacency.py "$XLEN x $YLEN" "$SAVENAME"
python src/scripts/generate_fuel_breaks.py "$XLEN x $YLEN" "$SAVENAME" "random"
python src/scripts/generate_fuel_breaks.py "$XLEN x $YLEN" "$SAVENAME" "domirank"
python src/scripts/generate_fuel_breaks.py "$XLEN x $YLEN" "$SAVENAME" "degree"
python src/scripts/generate_fuel_breaks.py "$XLEN x $YLEN" "$SAVENAME" "bonacich"
python src/scripts/simulate.py "$XLEN x $YLEN" "$SAVENAME" "random"
python src/scripts/simulate.py "$XLEN x $YLEN" "$SAVENAME" "domirank"
python src/scripts/simulate.py "$XLEN x $YLEN" "$SAVENAME" "degree"
python src/scripts/simulate.py "$XLEN x $YLEN" "$SAVENAME" "bonacich"


