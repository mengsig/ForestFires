#!/usr/bin/env python3
"""
Configuration Manager for Fire Simulation Framework

This module provides centralized configuration management for all fire simulation scripts.
All configuration parameters are loaded from simulation_config.yaml.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

# Add yaml support
try:
    import yaml
except ImportError:
    print("Error: PyYAML is required for configuration management.")
    print("Install it with: pip install PyYAML")
    sys.exit(1)

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Centralized configuration manager for fire simulation framework."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file. If None, searches for simulation_config.yaml
        """
        self.config_file = self._find_config_file(config_file)
        self.config = self._load_config()
        self._setup_environment()
        
    def _find_config_file(self, config_file: Optional[str] = None) -> Path:
        """Find the configuration file."""
        if config_file:
            config_path = Path(config_file)
            if config_path.exists():
                return config_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Search for simulation_config.yaml in common locations
        search_paths = [
            Path.cwd() / "simulation_config.yaml",
            Path(__file__).parent.parent.parent / "simulation_config.yaml",
            Path.cwd().parent / "simulation_config.yaml",
            Path("/") / "simulation_config.yaml",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
                
        raise FileNotFoundError(
            "simulation_config.yaml not found. Please ensure it exists in the project root."
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from: {self.config_file}")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_file}: {e}")
    
    def _setup_environment(self) -> None:
        """Setup environment variables based on configuration."""
        # Set up paths
        project_root = self.get_path('project_root')
        if not project_root:
            # Auto-detect project root
            project_root = self.config_file.parent
            self.config['paths']['project_root'] = str(project_root)
        
        # Set up performance settings
        perf = self.get_performance_config()
        if perf.get('numpy_num_threads'):
            os.environ['NUMPY_NUM_THREADS'] = str(perf['numpy_num_threads'])
        if perf.get('omp_num_threads'):
            os.environ['OMP_NUM_THREADS'] = str(perf['omp_num_threads'])
        
        # Set up random seed
        random_seed = self.get_advanced_setting('random_seed', 0)
        if random_seed > 0:
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            try:
                import numpy as np
                np.random.seed(random_seed)
            except ImportError:
                # Numpy not available, skip numpy random seed
                pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'simulation.grid_width')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration section."""
        return self.config.get('simulation', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration section."""
        return self.config.get('performance', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.config.get('logging', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration section."""
        return self.config.get('visualization', {})
    
    def get_fire_simulation_config(self) -> Dict[str, Any]:
        """Get fire simulation parameters."""
        return self.config.get('fire_simulation', {})
    
    def get_path(self, path_key: str) -> str:
        """Get path configuration."""
        return self.config.get('paths', {}).get(path_key, "")
    
    def get_paths_config(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self.config.get('paths', {})
    
    def get_raster_config(self) -> Dict[str, Any]:
        """Get raster data configuration."""
        return self.config.get('raster_data', {})
    
    def get_advanced_setting(self, setting_key: str, default: Any = None) -> Any:
        """Get advanced setting."""
        return self.config.get('advanced', {}).get(setting_key, default)
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get requirements configuration."""
        return self.config.get('requirements', {})
    
    # Convenience methods for commonly used values
    def get_grid_size(self) -> tuple[int, int]:
        """Get grid dimensions as (width, height)."""
        sim_config = self.get_simulation_config()
        return (sim_config.get('grid_width', 250), sim_config.get('grid_height', 250))
    
    def get_centralities(self) -> List[str]:
        """Get list of available centralities."""
        return self.get_simulation_config().get('centralities', ['domirank', 'random', 'degree', 'bonacich'])
    
    def get_fuel_break_percentages(self) -> List[int]:
        """Get list of fuel break percentages."""
        return self.get_simulation_config().get('fuel_break_percentages', [0, 5, 10, 15, 20, 25, 30])
    
    def get_num_parallel_simulations(self) -> int:
        """Get number of parallel simulations."""
        return self.get_simulation_config().get('num_parallel_simulations', 100)
    
    def get_max_parallel_jobs(self) -> int:
        """Get maximum parallel jobs (auto-detect if 0)."""
        max_jobs = self.get_performance_config().get('max_parallel_jobs', 0)
        if max_jobs == 0:
            import multiprocessing as mp
            max_jobs = max(1, mp.cpu_count() - 1)
        return max_jobs
    
    def get_time_steps(self, x_length: int, y_length: int) -> int:
        """Calculate time steps based on grid size."""
        sim_config = self.get_simulation_config()
        multiplier = sim_config.get('time_step_multiplier', 2500)
        divisor = sim_config.get('time_step_divisor', 400)
        try:
            import numpy as np
            return int(multiplier * np.sqrt(x_length * y_length) / divisor)
        except ImportError:
            # Fall back to math.sqrt if numpy is not available
            import math
            return int(multiplier * math.sqrt(x_length * y_length) / divisor)
    
    def get_cube_resolution(self) -> tuple[int, int, int]:
        """Get space-time cube resolution."""
        fire_config = self.get_fire_simulation_config()
        cube_res = fire_config.get('cube_resolution', {})
        return (
            cube_res.get('band_duration', 60),
            cube_res.get('cell_height', 30),
            cube_res.get('cell_width', 30)
        )
    
    def get_fuel_moisture_values(self) -> Dict[str, float]:
        """Get fuel moisture values."""
        fire_config = self.get_fire_simulation_config()
        return fire_config.get('fuel_moisture', {
            'dead_1hr': 0.10,
            'dead_10hr': 0.25,
            'dead_100hr': 0.50,
            'live_herbaceous': 0.90,
            'live_woody': 0.60,
            'foliar': 0.90
        })
    
    def get_project_root(self) -> Path:
        """Get project root directory."""
        root = self.get_path('project_root')
        return Path(root) if root else Path.cwd()
    
    def get_results_dir(self) -> Path:
        """Get results directory."""
        return self.get_project_root() / self.get_path('results_dir')
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory."""
        temp_dir = self.get_path('temp_dir')
        if not temp_dir:
            import tempfile
            temp_dir = tempfile.gettempdir()
        
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        return temp_path
    
    def create_simulation_name(self, base_name: Optional[str] = None) -> str:
        """Create a simulation name with timestamp."""
        if not base_name:
            base_name = self.get_simulation_config().get('default_name', 'fire_sim')
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp}"
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check required sections
        required_sections = ['simulation', 'performance', 'paths', 'fire_simulation']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required configuration section: {section}")
        
        # Validate simulation parameters
        sim_config = self.get_simulation_config()
        if sim_config.get('grid_width', 0) <= 0:
            issues.append("Invalid grid_width: must be positive")
        if sim_config.get('grid_height', 0) <= 0:
            issues.append("Invalid grid_height: must be positive")
        if sim_config.get('num_parallel_simulations', 0) <= 0:
            issues.append("Invalid num_parallel_simulations: must be positive")
        
        # Validate paths
        project_root = self.get_project_root()
        if not project_root.exists():
            issues.append(f"Project root directory does not exist: {project_root}")
        
        return issues
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        log_config = self.get_logging_config()
        
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[]
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Add colored output if enabled
        if log_config.get('colored_output', True):
            try:
                import colorlog
                formatter = colorlog.ColoredFormatter(
                    '%(log_color)s' + format_str,
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'blue',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                )
                console_handler.setFormatter(formatter)
            except ImportError:
                # Fall back to regular formatter if colorlog not available
                formatter = logging.Formatter(format_str)
                console_handler.setFormatter(formatter)
        else:
            formatter = logging.Formatter(format_str)
            console_handler.setFormatter(formatter)
        
        # Add file handler if specified
        log_file = log_config.get('log_file', '')
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(format_str)
            file_handler.setFormatter(file_formatter)
            logging.getLogger().addHandler(file_handler)
        
        logging.getLogger().addHandler(console_handler)
        
        return logging.getLogger(__name__)


# Global configuration instance
_config = None

def get_config(config_file: Optional[str] = None) -> ConfigurationManager:
    """
    Get global configuration instance.
    
    Args:
        config_file: Path to configuration file (only used on first call)
        
    Returns:
        ConfigurationManager instance
    """
    global _config
    if _config is None:
        _config = ConfigurationManager(config_file)
    return _config

def reset_config():
    """Reset global configuration instance (mainly for testing)."""
    global _config
    _config = None

# Convenience functions for common operations
def get_grid_size() -> tuple[int, int]:
    """Get grid dimensions."""
    return get_config().get_grid_size()

def get_centralities() -> List[str]:
    """Get available centralities."""
    return get_config().get_centralities()

def get_fuel_break_percentages() -> List[int]:
    """Get fuel break percentages."""
    return get_config().get_fuel_break_percentages()

def get_project_root() -> Path:
    """Get project root directory."""
    return get_config().get_project_root()

def get_results_dir() -> Path:
    """Get results directory."""
    return get_config().get_results_dir()

def setup_logging() -> logging.Logger:
    """Setup logging from configuration."""
    return get_config().setup_logging()