# Installation

If you are using **arch** linux, you can install directly via the command:

```bash
bash install.sh --IUseArchBtw
```

If you are using **debian** with the *apt* package-manager, you can install via:

```bash
bash install.sh 
```

Otherwise, on any **OS** you can install by first ensuring that you are using
python version **~Python3.10**. Thereafter, execute the following commands:

```bash
git clone git@github.com:pyregence/pyretechnics.git
git clone git@github.com:mengsig/DomiRank.git
python -m venv forestfires
source forestfires/bin/activate
pip install -e DomiRank/.
pip install -r requirements.txt
cd pyregence
python setup.py install
cd ..
```

# Running Simulations and Creating Fire Breaks
The *run.sh* script provides a bash user interface for the user to change various components of their simulation:
1. ```XLEN``` - grid size in horizontal dimension.
2. ```YLEN``` - grid size in vertical dimension.
3. ```SAVENAME``` - results will be stored in *src/results/<savename>*.
4. ```PERC``` - a vector of fuel-break fractions desired to investigate.
5. ```CENTRALITIES``` - a vector of the names of the centralities desired to use for creating fuel-breaks. These are found / should be placed in *src/utils/centralityUtils.py*

Once you have created your desired configuration, simply run.

```bash
bash run.sh
```

And all the results should be stored in the *src/results/<savename>/* folder.

# Running Aggregated Simulations and Creating Fire Breaks
To run *run.sh* many times (default 100), please use the *fullrun.sh* bash script. Similarly, the arguments it takes are:
1. ```XLEN``` - grid size in horizontal dimension.
2. ```YLEN``` - grid size in vertical dimension.
3. ```SAVENAME``` - results will be stored in *src/results/<savename>*.
4. ```PERC``` - a vector of fuel-break fractions desired to investigate.
5. ```CENTRALITIES``` - a vector of the names of the centralities desired to use for creating fuel-breaks. These are found / should be placed in *src/utils/centralityUtils.py*
6. ```NUM_SIMULATIONS``` - an integer value that changes the total number of simulations considered in the aggregation of results.

Once you have created your desired configuration, simply run.

```bash
bash fullrun.sh
```


# Reproducing exactly paper results:
If you are using a *debian* distribution with apt pkg-manager, then simply run:

```bash
bash reproduce.sh
```

If you are using *arch* with pacman and yay as a pkg-managers, then simply run:

```bash
bash reproduce.sh --IUseArchBtw
```

If you are on another OS: then please install the repository by following the instructions above, and then simply run:
```bash
bash run.sh & bash fullrun.sh
```

Your **results** will be in *results/* in two folders named *results/250_8neighbors* and *results/full_250_8neighbors* for the individual and aggregated results respectively.


### Enjoy (;


### By - Marcus Engsig

