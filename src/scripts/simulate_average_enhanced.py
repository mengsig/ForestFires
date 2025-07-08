#!/usr/bin/env python3
"""
Enhanced Fire Simulation with Parallel Processing

This script simulates fire spread modeling with parallelized averaging over multiple runs.
Key improvements:
- Parallel processing of simulation runs using multiprocessing
- Better code organization and error handling
- Reduced memory usage through optimized data structures
- Type hints and comprehensive documentation
"""

import os
import sys
import shutil
import time
import logging
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import numpy as np
from pprint import pprint

# Pyretechnics imports
from pyretechnics.space_time_cube import SpaceTimeCube
import pyretechnics.eulerian_level_set as els

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from src.utils.loadingUtils import load_raster, convert_to_cube
from src.utils.plottingUtils import save_matrix_as_heatmap
from src.utils.parsingUtils import parse_args
from src.utils.configManager import get_config


class FireSimulationConfig:
    """Configuration class for fire simulation parameters."""
    
    def __init__(self, x_length: int, y_length: int, save_name: str, 
                 centrality: str, fuel_break_fraction: int):
        # Load global configuration
        self.config_manager = get_config()
        
        # Basic parameters
        self.x_length = x_length
        self.y_length = y_length
        self.save_name = save_name
        self.centrality = centrality
        self.fuel_break_fraction = fuel_break_fraction
        
        # Load parameters from configuration
        self.time_steps = self.config_manager.get_time_steps(x_length, y_length)
        sim_config = self.config_manager.get_simulation_config()
        self.x_offset = sim_config.get('x_offset', 100)
        self.y_offset = sim_config.get('y_offset', 100)
        self.num_simulations = self.config_manager.get_num_parallel_simulations()
        
        # Setup paths using configuration
        results_dir = self.config_manager.get_path('results_dir')
        self.savedir = Path(results_dir) / save_name
        self.fuel_breaks_file = self.savedir / f"{centrality}_{fuel_break_fraction}.txt"
        self.fuel_breaks_img = self.savedir / f"{centrality}_{fuel_break_fraction}.png"
        
    def create_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        if self.fuel_breaks_file.exists():
            self.savedir = Path(str(self.fuel_breaks_file).removesuffix(".txt"))
        self.savedir.mkdir(parents=True, exist_ok=True)


class RasterDataManager:
    """Manages loading and preprocessing of raster data."""
    
    def __init__(self, config: FireSimulationConfig):
        self.config = config
        self.raster_data = {}
        self.fuel_breaks = None
        
    def load_all_rasters(self) -> Dict[str, np.ndarray]:
        """Load all required raster data."""
        logger.info("Loading raster data...")
        
        x_subset = (self.config.x_offset, self.config.x_offset + self.config.x_length)
        y_subset = (self.config.y_offset, self.config.y_offset + self.config.y_length)
        
        raster_types = ["slp", "asp", "dem", "cc", "cbd", "cbh", "ch", "fbfm"]
        
        for raster_type in raster_types:
            try:
                self.raster_data[raster_type] = load_raster(raster_type, x_subset, y_subset)
                logger.debug(f"Loaded {raster_type} raster: {self.raster_data[raster_type].shape}")
            except Exception as e:
                logger.error(f"Failed to load {raster_type} raster: {e}")
                raise
                
        return self.raster_data
    
    def apply_fuel_breaks(self) -> None:
        """Apply fuel breaks to the raster data."""
        if self.config.fuel_breaks_file.exists():
            logger.info("Applying fuel breaks...")
            self.fuel_breaks = np.loadtxt(self.config.fuel_breaks_file).astype(bool)
            
            # Get fuel break values from configuration
            config_manager = self.config.config_manager
            fire_config = config_manager.get_fire_simulation_config()
            fuel_break_config = fire_config.get('fuel_break', {})
            
            # Apply fuel breaks using configuration values
            self.raster_data["fbfm"][self.fuel_breaks] = fuel_break_config.get('fuel_model_value', 91)
            self.raster_data["cc"][self.fuel_breaks] = fuel_break_config.get('canopy_cover_value', 0)
            self.raster_data["cbd"][self.fuel_breaks] = fuel_break_config.get('canopy_bulk_density_value', 0)
            
            # Clean up files
            rel_img_name = self.config.fuel_breaks_img.name
            shutil.move(str(self.config.fuel_breaks_img), 
                       self.config.savedir / rel_img_name)
            self.config.fuel_breaks_file.unlink()
            
            logger.info(f"Applied fuel breaks to {np.sum(self.fuel_breaks)} cells")


class SpaceTimeCubeBuilder:
    """Builds space-time cubes for simulation."""
    
    def __init__(self, config: FireSimulationConfig, raster_data: Dict[str, np.ndarray]):
        self.config = config
        self.raster_data = raster_data
        
    def build_cubes(self) -> Dict[str, SpaceTimeCube]:
        """Build all space-time cubes for simulation."""
        logger.info("Building space-time cubes...")
        
        rows, cols = self.raster_data["slp"].shape
        cube_shape = (self.config.time_steps, rows, cols)
        
        # Convert rasters to cubes
        cubes = {}
        cube_mappings = {
            "slope": ("slp", "slp"),
            "aspect": ("asp", "asp"),
            "fuel_model": ("fbfm", "fbfm"),
            "canopy_cover": ("cc", "cc"),
            "canopy_height": ("ch", "ch"),
            "canopy_base_height": ("cbh", "cbh"),
            "canopy_bulk_density": ("cbd", "cbd"),
        }
        
        for cube_name, (raster_key, datatype) in cube_mappings.items():
            cube_data = convert_to_cube(self.raster_data[raster_key], 
                                      self.config.time_steps, datatype=datatype)
            cubes[cube_name] = SpaceTimeCube(cube_shape, cube_data)
        
        # Add constant cubes from configuration
        fire_config = self.config.get_fire_simulation_config()
        fuel_moisture = fire_config.get('fuel_moisture', {})
        wind_config = fire_config.get('wind', {})
        adjustments = fire_config.get('adjustments', {})
        
        constant_cubes = {
            "wind_speed_10m": wind_config.get('speed_10m', 0),
            "upwind_direction": wind_config.get('upwind_direction', 0),
            "fuel_moisture_dead_1hr": fuel_moisture.get('dead_1hr', 0.10),
            "fuel_moisture_dead_10hr": fuel_moisture.get('dead_10hr', 0.25),
            "fuel_moisture_dead_100hr": fuel_moisture.get('dead_100hr', 0.50),
            "fuel_moisture_live_herbaceous": fuel_moisture.get('live_herbaceous', 0.90),
            "fuel_moisture_live_woody": fuel_moisture.get('live_woody', 0.60),
            "foliar_moisture": fuel_moisture.get('foliar', 0.90),
            "fuel_spread_adjustment": adjustments.get('fuel_spread', 1.0),
            "weather_spread_adjustment": adjustments.get('weather_spread', 1.0),
        }
        
        for name, value in constant_cubes.items():
            cubes[name] = SpaceTimeCube(cube_shape, value)
            
        logger.info(f"Built {len(cubes)} space-time cubes")
        return cubes


def run_single_simulation(simulation_params: Tuple[Dict[str, SpaceTimeCube], 
                                                  Tuple[int, int, int], 
                                                  Tuple[int, int], 
                                                  int]) -> Dict[str, np.ndarray]:
    """Run a single fire simulation."""
    space_time_cubes, cube_shape, grid_size, sim_id = simulation_params
    x_length, y_length = grid_size
    
    # Load configuration for simulation parameters
    config = get_config()
    fire_config = config.get_fire_simulation_config()
    
    # Random ignition point
    x_cord = np.random.randint(0, x_length)
    y_cord = np.random.randint(0, y_length)
    
    spread_state = els.SpreadState(cube_shape).ignite_cell((x_cord, y_cord))
    
    # Get cube resolution from configuration
    cube_resolution = config.get_cube_resolution()
    
    # Get simulation timing from configuration
    start_time = fire_config.get('start_time', 0)
    max_duration_factor = fire_config.get('max_duration_factor', 0.75)
    max_duration = int(cube_shape[0] * max_duration_factor) * 60  # minutes
    
    # Get fire model from configuration
    surface_model = fire_config.get('surface_lw_ratio_model', 'rothermel')
    
    try:
        fire_spread_results = els.spread_fire_with_phi_field(
            space_time_cubes,
            spread_state,
            cube_resolution,
            start_time,
            max_duration,
            surface_lw_ratio_model=surface_model,
        )
        
        spread_state = fire_spread_results["spread_state"]
        return spread_state.get_full_matrices()
        
    except Exception as e:
        logger.error(f"Simulation {sim_id} failed: {e}")
        return None


class ParallelFireSimulator:
    """Main class for running parallel fire simulations."""
    
    def __init__(self, config: FireSimulationConfig):
        self.config = config
        self.raster_manager = RasterDataManager(config)
        self.results = None
        
    def setup(self) -> None:
        """Setup the simulator by loading data and building cubes."""
        self.config.create_output_directory()
        
        # Load and process raster data
        raster_data = self.raster_manager.load_all_rasters()
        self.raster_manager.apply_fuel_breaks()
        
        # Build space-time cubes
        cube_builder = SpaceTimeCubeBuilder(self.config, raster_data)
        self.space_time_cubes = cube_builder.build_cubes()
        
        rows, cols = raster_data["slp"].shape
        self.cube_shape = (self.config.time_steps, rows, cols)
        self.grid_size = (self.config.x_length, self.config.y_length)
        
    def run_parallel_simulations(self) -> Dict[str, np.ndarray]:
        """Run multiple simulations in parallel and aggregate results."""
        logger.info(f"Running {self.config.num_simulations} parallel simulations...")
        
        # Determine optimal number of processes from configuration
        config_manager = self.config.config_manager
        max_processes = config_manager.get_max_parallel_jobs()
        num_processes = min(max_processes, self.config.num_simulations)
        logger.info(f"Using {num_processes} processes (max configured: {max_processes})")
        
        # Prepare simulation parameters
        simulation_params = [
            (self.space_time_cubes, self.cube_shape, self.grid_size, i)
            for i in range(self.config.num_simulations)
        ]
        
        # Initialize result accumulators
        aggregated_results = None
        successful_simulations = 0
        
        start_time = time.perf_counter()
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_sim = {
                executor.submit(run_single_simulation, params): i 
                for i, params in enumerate(simulation_params)
            }
            
            for future in as_completed(future_to_sim):
                sim_id = future_to_sim[future]
                try:
                    result = future.result()
                    if result is not None:
                        if aggregated_results is None:
                            # Initialize with first successful result
                            aggregated_results = {key: np.array(value) for key, value in result.items()}
                        else:
                            # Accumulate results
                            for key in aggregated_results.keys():
                                if key != "fire_type":  # fire_type is not averaged
                                    aggregated_results[key] += result[key]
                        
                        successful_simulations += 1
                        if successful_simulations % 10 == 0:
                            logger.info(f"Completed {successful_simulations}/{self.config.num_simulations} simulations")
                            
                except Exception as e:
                    logger.error(f"Simulation {sim_id} failed: {e}")
        
        end_time = time.perf_counter()
        logger.info(f"Completed {successful_simulations} simulations in {end_time - start_time:.2f} seconds")
        
        if aggregated_results is None or successful_simulations == 0:
            raise RuntimeError("No simulations completed successfully")
        
        # Average the results (except fire_type)
        for key in aggregated_results.keys():
            if key != "fire_type":
                aggregated_results[key] = aggregated_results[key] / successful_simulations
        
        # Set fire_type to the last successful simulation (or could be majority vote)
        logger.info(f"Averaged results from {successful_simulations} successful simulations")
        return aggregated_results
    
    def save_results(self, output_matrices: Dict[str, np.ndarray]) -> None:
        """Save simulation results and statistics."""
        logger.info("Saving results...")
        
        num_burned_cells = np.count_nonzero(output_matrices["fire_type"])
        acres_burned = num_burned_cells / 4.5
        
        logger.info(f"[SIMULATE-{self.config.centrality}]: Acres Burned: {acres_burned}")
        
        # Save statistics
        self._save_statistics(output_matrices, acres_burned)
        
        # Apply fuel breaks to output matrices for visualization
        if self.raster_manager.fuel_breaks is not None:
            self._apply_fuel_breaks_to_output(output_matrices)
        
        # Save heatmaps
        self._save_heatmaps(output_matrices)
        
    def _save_statistics(self, output_matrices: Dict[str, np.ndarray], acres_burned: float) -> None:
        """Save detailed statistics to file."""
        burned_cells = output_matrices["fire_type"] > 0
        
        def get_array_stats(array, use_burn_scar_mask=True):
            array_values = array[burned_cells] if use_burn_scar_mask else array
            if len(array_values) > 0:
                return {
                    "Min": np.min(array_values),
                    "Max": np.max(array_values),
                    "Mean": np.mean(array_values),
                    "Stdev": np.std(array_values),
                }
            return {"Min": "No Data", "Max": "No Data", "Mean": "No Data", "Stdev": "No Data"}
        
        stats_path = self.config.savedir / "stats.txt"
        with open(stats_path, "w") as f:
            f.write("Enhanced Fire Behavior Simulation Results\n" + "=" * 60 + "\n\n")
            f.write(f"Simulation Configuration:\n")
            f.write(f"  Grid Size: {self.config.x_length}x{self.config.y_length}\n")
            f.write(f"  Centrality: {self.config.centrality}\n")
            f.write(f"  Fuel Break Fraction: {self.config.fuel_break_fraction}%\n")
            f.write(f"  Number of Simulations: {self.config.num_simulations}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Acres Burned: {acres_burned:.2f}\n\n")
            
            # Detailed statistics for each metric
            metrics = [
                ("Phi", "phi", "phi <= 0: burned, phi > 0: unburned", False),
                ("Fire Type", "fire_type", "0=unburned, 1=surface, 2=passive_crown, 3=active_crown", True),
                ("Spread Rate", "spread_rate", "m/min", True),
                ("Spread Direction", "spread_direction", "degrees clockwise from North", True),
                ("Fireline Intensity", "fireline_intensity", "kW/m", True),
                ("Flame Length", "flame_length", "meters", True),
                ("Time of Arrival", "time_of_arrival", "minutes", True),
            ]
            
            for name, key, units, use_mask in metrics:
                f.write(f"{name} ({units}):\n")
                stats = get_array_stats(output_matrices[key], use_mask)
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value}\n")
                f.write("\n")
    
    def _apply_fuel_breaks_to_output(self, output_matrices: Dict[str, np.ndarray]) -> None:
        """Apply fuel breaks to output matrices for visualization."""
        fuel_breaks = self.raster_manager.fuel_breaks
        output_matrices["phi"][fuel_breaks] = np.nan
        output_matrices["fire_type"][fuel_breaks] = 0
        output_matrices["spread_rate"][fuel_breaks] = np.nan
        output_matrices["spread_direction"][fuel_breaks] = np.nan
        output_matrices["fireline_intensity"][fuel_breaks] = np.inf
        output_matrices["flame_length"][fuel_breaks] = np.nan
        output_matrices["time_of_arrival"][fuel_breaks] = np.nan
    
    def _save_heatmaps(self, output_matrices: Dict[str, np.ndarray]) -> None:
        """Save heatmap visualizations."""
        # Prepare fireline intensity for visualization
        output_matrices["fireline_intensity"] += 0.01
        
        # Get canopy coverage for background
        cc_data = self.raster_manager.raster_data["cc"]
        cc_cube = convert_to_cube(cc_data, self.config.time_steps, datatype="cc")
        
        heatmap_configs = [
            {
                "matrix": output_matrices["phi"],
                "colors": "plasma",
                "units": "phi <= 0: burned, phi > 0: unburned",
                "title": "Phi",
                "filename": str(self.config.savedir / "els_phi.png"),
            },
            {
                "matrix": output_matrices["fire_type"],
                "colors": "viridis",
                "units": "0=unburned, 1=surface, 2=passive_crown, 3=active_crown",
                "title": "Fire Type",
                "filename": str(self.config.savedir / "els_fire_type.png"),
                "vmin": 0,
                "vmax": 3,
                "ticks": [0, 1, 2, 3],
            },
            {
                "matrix": output_matrices["spread_rate"],
                "colors": "hot",
                "units": "m/min",
                "title": "Spread Rate",
                "filename": str(self.config.savedir / "els_spread_rate.png"),
            },
            {
                "matrix": output_matrices["spread_direction"],
                "colors": "viridis",
                "units": "degrees clockwise from North",
                "title": "Spread Direction",
                "filename": str(self.config.savedir / "els_spread_direction.png"),
                "vmin": 0,
                "vmax": 360,
                "ticks": [0, 45, 90, 135, 180, 225, 270, 315, 360]
            },
            {
                "matrix": output_matrices["fireline_intensity"],
                "colors": "hot",
                "units": "kW/m",
                "title": "Fireline Intensity",
                "filename": str(self.config.savedir / "els_fireline_intensity.png"),
                "vmin": output_matrices['fireline_intensity'].min(),
                "vmax": output_matrices['fireline_intensity'].max(),
                "norm": True
            },
            {
                "matrix": output_matrices["flame_length"],
                "colors": "hot",
                "units": "meters",
                "title": "Flame Length",
                "filename": str(self.config.savedir / "els_flame_length.png"),
            },
            {
                "matrix": np.flip(cc_cube[0, :, :], axis=0),
                "colors": "Greens",
                "units": "coverage",
                "title": "Canopy Coverage",
                "filename": str(self.config.savedir / "els_canopy_coverage.png"),
            },
            {
                "matrix": output_matrices["time_of_arrival"],
                "colors": "viridis",
                "units": "minutes",
                "vmin": 1,
                "vmax": np.max(output_matrices["time_of_arrival"][output_matrices["time_of_arrival"] > 0]),
                "title": "Time of Arrival",
                "filename": str(self.config.savedir / "els_time_of_arrival.png"),
            },
        ]
        
        for config in heatmap_configs:
            try:
                save_matrix_as_heatmap(**config)
            except Exception as e:
                logger.error(f"Failed to save heatmap {config['title']}: {e}")
    
    def run(self) -> None:
        """Run the complete simulation workflow."""
        logger.info("Starting enhanced fire simulation...")
        
        try:
            self.setup()
            output_matrices = self.run_parallel_simulations()
            self.save_results(output_matrices)
            logger.info("Simulation completed successfully!")
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise


def main():
    """Main entry point for the enhanced fire simulation."""
    try:
        # Parse command line arguments
        x_length, y_length, save_name, centrality, fuel_break_fraction = parse_args()
        
        # Create configuration
        config = FireSimulationConfig(x_length, y_length, save_name, centrality, fuel_break_fraction)
        
        # Run simulation
        simulator = ParallelFireSimulator(config)
        simulator.run()
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()