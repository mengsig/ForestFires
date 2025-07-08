# Quick Start Guide - Enhanced Fire Simulation

## 🎯 Issues Fixed

✅ **Resolved the temporary directory issue** in `meanfullrun_enhanced.sh`  
✅ **Implemented centralized configuration** system as requested  
✅ **All scripts now read from `simulation_config.yaml`** for parameters  

## 🚀 Getting Started

### 1. Install Dependencies

First, make sure you have PyYAML installed:

```bash
pip install PyYAML
# or install all enhanced dependencies
pip install -r requirements_enhanced.txt
```

### 2. Configure Your Simulation

Edit `simulation_config.yaml` to customize your simulation parameters:

```yaml
simulation:
  grid_width: 250          # Adjust grid size
  grid_height: 250
  num_parallel_simulations: 100  # Number of parallel runs
  centralities:            # Available centrality measures
    - "domirank"
    - "random" 
    - "degree"
    - "bonacich"
  fuel_break_percentages:  # Percentages to test
    - 0
    - 5
    - 10
    - 15
    - 20
    - 25
    - 30

performance:
  max_parallel_jobs: 0     # 0 = auto-detect CPU cores
```

### 3. Run Enhanced Simulation

```bash
# Run the enhanced pipeline (uses configuration from YAML)
bash meanfullrun_enhanced.sh
```

### 4. Run Single Enhanced Simulation

```bash
# Run a single simulation (also uses configuration)
python3 src/scripts/simulate_average_enhanced.py 250x250 my_simulation domirank 15
```

## 📊 Key Improvements

### Centralized Configuration
- **All parameters** are now in `simulation_config.yaml`
- **No more hardcoded values** scattered across scripts
- **Easy to modify** simulation settings in one place
- **Environment-specific** configurations possible

### Enhanced Performance
- **Fixed shell script issues** - no more temp directory errors
- **Intelligent process management** - automatic CPU detection
- **Parallel simulation execution** within `simulate_average_enhanced.py`
- **Better memory management** and error handling

### Configuration Examples

#### Quick Test Configuration
```yaml
simulation:
  grid_width: 100
  grid_height: 100
  num_parallel_simulations: 25
performance:
  max_parallel_jobs: 2
```

#### High-Performance Configuration
```yaml
simulation:
  grid_width: 500
  grid_height: 500
  num_parallel_simulations: 200
performance:
  max_parallel_jobs: 16
```

#### Custom Fire Parameters
```yaml
fire_simulation:
  fuel_moisture:
    dead_1hr: 0.08      # Drier conditions
    dead_10hr: 0.20
    live_herbaceous: 0.80
  wind:
    speed_10m: 5        # 5 m/s wind speed
    upwind_direction: 45 # Northeast wind
```

## 🔧 Configuration Options

### Simulation Parameters
- `grid_width/height`: Grid dimensions
- `num_parallel_simulations`: Number of simulations to average
- `centralities`: Available centrality measures
- `fuel_break_percentages`: Percentages to test

### Performance Settings
- `max_parallel_jobs`: Number of parallel processes (0 = auto)
- `memory_limit_mb`: Memory limit per process
- `enable_parallel_processing`: Enable/disable parallelization

### Fire Model Parameters
- `fuel_moisture`: Moisture content for different fuel types
- `wind`: Wind speed and direction
- `cube_resolution`: Spatial and temporal resolution
- `fuel_break`: Fuel break model parameters

### Paths and Output
- `results_dir`: Where to save results
- `temp_dir`: Temporary file location
- `raster_data_dir`: Input raster data location

## 📁 File Structure

```
project/
├── simulation_config.yaml          # ← Main configuration file
├── meanfullrun_enhanced.sh         # ← Enhanced shell pipeline
├── src/
│   ├── scripts/
│   │   └── simulate_average_enhanced.py  # ← Enhanced simulation script
│   └── utils/
│       └── configManager.py        # ← Configuration manager
├── requirements_enhanced.txt       # ← Enhanced dependencies
└── README_ENHANCED.md              # ← Full documentation
```

## 🚨 Troubleshooting

### Configuration Issues
```bash
# Test configuration loading
python3 -c "from src.utils.configManager import get_config; print('✅ Config OK')"
```

### Missing Dependencies
```bash
# Install missing dependencies
pip install PyYAML numpy matplotlib rasterio
```

### Shell Script Issues
```bash
# Check shell script configuration loading
bash -c "source meanfullrun_enhanced.sh; echo 'Grid: ${XLEN}x${YLEN}'"
```

### Performance Issues
```bash
# Reduce parallel jobs if running out of memory
# Edit simulation_config.yaml:
performance:
  max_parallel_jobs: 2
  memory_limit_mb: 2048
```

## 💡 Tips

1. **Start small**: Test with smaller grid sizes first
2. **Monitor resources**: Watch CPU and memory usage during runs
3. **Customize gradually**: Modify one parameter at a time
4. **Check logs**: Enhanced logging provides detailed progress information
5. **Use validation**: The configuration system validates parameters automatically

## 🎉 What's New

- ✅ **No more temp directory errors**
- ✅ **Centralized configuration in YAML**
- ✅ **All scripts use same configuration**
- ✅ **Better error handling and logging**
- ✅ **Automatic CPU detection**
- ✅ **Configurable fire model parameters**
- ✅ **Enhanced documentation and validation**

## 🚀 Next Steps

1. Test the system with your configuration
2. Adjust parameters in `simulation_config.yaml` as needed
3. Run benchmarks to compare performance
4. Explore advanced configuration options
5. Check the full documentation in `README_ENHANCED.md`

---

**Happy simulating! 🔥**