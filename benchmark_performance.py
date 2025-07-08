#!/usr/bin/env python3
"""
Performance Benchmarking Tool for Fire Simulation Enhancement

This script compares the performance of the original and enhanced fire simulation
implementations to quantify the improvements.
"""

import os
import sys
import time
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PerformanceBenchmark:
    """Benchmarking suite for fire simulation performance comparison."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {}
        self.temp_dir = None
        
    def setup_temp_environment(self):
        """Setup temporary environment for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="fire_sim_benchmark_")
        print(f"Created temporary directory: {self.temp_dir}")
        
    def cleanup_temp_environment(self):
        """Clean up temporary environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def run_original_simulation(self, grid_size: str, save_name: str, 
                              centrality: str, fuel_fraction: int) -> Dict:
        """Run the original simulation implementation."""
        print(f"Running original simulation: {centrality} @ {fuel_fraction}%")
        
        start_time = time.perf_counter()
        
        try:
            # First generate fuel breaks
            subprocess.run([
                "python3", "src/scripts/generate_fuel_breaks.py",
                grid_size, save_name, centrality
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            # Run original simulation
            result = subprocess.run([
                "python3", "src/scripts/simulate_average.py",
                grid_size, save_name, centrality, str(fuel_fraction)
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            return {
                "success": True,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            return {
                "success": False,
                "duration": duration,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def run_enhanced_simulation(self, grid_size: str, save_name: str,
                               centrality: str, fuel_fraction: int) -> Dict:
        """Run the enhanced simulation implementation."""
        print(f"Running enhanced simulation: {centrality} @ {fuel_fraction}%")
        
        start_time = time.perf_counter()
        
        try:
            # First generate fuel breaks
            subprocess.run([
                "python3", "src/scripts/generate_fuel_breaks.py",
                grid_size, save_name, centrality
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            # Run enhanced simulation
            result = subprocess.run([
                "python3", "src/scripts/simulate_average_enhanced.py",
                grid_size, save_name, centrality, str(fuel_fraction)
            ], check=True, capture_output=True, text=True, cwd=self.project_root)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            return {
                "success": True,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            return {
                "success": False,
                "duration": duration,
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def run_benchmark_suite(self, test_cases: List[Tuple[str, str, str, int]], 
                           num_runs: int = 3) -> Dict:
        """Run complete benchmark suite."""
        print(f"Running benchmark suite with {len(test_cases)} test cases, {num_runs} runs each")
        
        results = {
            "original": {},
            "enhanced": {},
            "comparison": {}
        }
        
        for i, (grid_size, save_name, centrality, fuel_fraction) in enumerate(test_cases):
            test_id = f"{grid_size}_{centrality}_{fuel_fraction}"
            print(f"\nTest case {i+1}/{len(test_cases)}: {test_id}")
            
            # Run original implementation
            original_times = []
            for run in range(num_runs):
                print(f"  Original run {run+1}/{num_runs}")
                result = self.run_original_simulation(grid_size, f"{save_name}_orig_{run}", 
                                                    centrality, fuel_fraction)
                if result["success"]:
                    original_times.append(result["duration"])
                else:
                    print(f"    Original run failed: {result.get('error', 'Unknown error')}")
            
            # Run enhanced implementation
            enhanced_times = []
            for run in range(num_runs):
                print(f"  Enhanced run {run+1}/{num_runs}")
                result = self.run_enhanced_simulation(grid_size, f"{save_name}_enh_{run}", 
                                                     centrality, fuel_fraction)
                if result["success"]:
                    enhanced_times.append(result["duration"])
                else:
                    print(f"    Enhanced run failed: {result.get('error', 'Unknown error')}")
            
            # Store results
            results["original"][test_id] = {
                "times": original_times,
                "mean": np.mean(original_times) if original_times else None,
                "std": np.std(original_times) if original_times else None,
                "success_rate": len(original_times) / num_runs
            }
            
            results["enhanced"][test_id] = {
                "times": enhanced_times,
                "mean": np.mean(enhanced_times) if enhanced_times else None,
                "std": np.std(enhanced_times) if enhanced_times else None,
                "success_rate": len(enhanced_times) / num_runs
            }
            
            # Calculate comparison metrics
            if original_times and enhanced_times:
                speedup = np.mean(original_times) / np.mean(enhanced_times)
                results["comparison"][test_id] = {
                    "speedup": speedup,
                    "improvement_percent": (speedup - 1) * 100,
                    "time_saved": np.mean(original_times) - np.mean(enhanced_times)
                }
                print(f"  Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% improvement)")
            else:
                results["comparison"][test_id] = {
                    "speedup": None,
                    "improvement_percent": None,
                    "time_saved": None
                }
                print("  Unable to calculate speedup due to failures")
        
        return results
    
    def generate_performance_report(self, results: Dict, output_dir: str):
        """Generate comprehensive performance report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_path / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary statistics
        self._generate_summary_stats(results, output_path)
        
        # Create visualizations
        self._generate_performance_plots(results, output_path)
        
        # Create detailed report
        self._generate_detailed_report(results, output_path)
        
        print(f"\nPerformance report generated in: {output_path}")
    
    def _generate_summary_stats(self, results: Dict, output_path: Path):
        """Generate summary statistics."""
        summary = {
            "total_tests": len(results["comparison"]),
            "successful_comparisons": sum(1 for comp in results["comparison"].values() 
                                        if comp["speedup"] is not None),
            "average_speedup": None,
            "max_speedup": None,
            "min_speedup": None,
            "total_time_saved": None
        }
        
        valid_speedups = [comp["speedup"] for comp in results["comparison"].values() 
                         if comp["speedup"] is not None]
        
        if valid_speedups:
            summary["average_speedup"] = np.mean(valid_speedups)
            summary["max_speedup"] = np.max(valid_speedups)
            summary["min_speedup"] = np.min(valid_speedups)
            
            valid_time_saved = [comp["time_saved"] for comp in results["comparison"].values() 
                               if comp["time_saved"] is not None]
            summary["total_time_saved"] = sum(valid_time_saved)
        
        with open(output_path / "summary_stats.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nSummary Statistics:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Successful comparisons: {summary['successful_comparisons']}")
        if summary["average_speedup"]:
            print(f"  Average speedup: {summary['average_speedup']:.2f}x")
            print(f"  Best speedup: {summary['max_speedup']:.2f}x")
            print(f"  Worst speedup: {summary['min_speedup']:.2f}x")
            print(f"  Total time saved: {summary['total_time_saved']:.1f} seconds")
    
    def _generate_performance_plots(self, results: Dict, output_path: Path):
        """Generate performance visualization plots."""
        # Extract data for plotting
        test_ids = []
        speedups = []
        original_times = []
        enhanced_times = []
        
        for test_id, comparison in results["comparison"].items():
            if comparison["speedup"] is not None:
                test_ids.append(test_id)
                speedups.append(comparison["speedup"])
                original_times.append(results["original"][test_id]["mean"])
                enhanced_times.append(results["enhanced"][test_id]["mean"])
        
        if not test_ids:
            print("No valid data for plotting")
            return
        
        # Create speedup comparison plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(test_ids)), speedups, color='steelblue', alpha=0.7)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
        plt.xlabel('Test Cases')
        plt.ylabel('Speedup Factor')
        plt.title('Performance Speedup by Test Case')
        plt.xticks(range(len(test_ids)), test_ids, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=8)
        
        plt.subplot(2, 2, 2)
        x = np.arange(len(test_ids))
        width = 0.35
        plt.bar(x - width/2, original_times, width, label='Original', alpha=0.7, color='orange')
        plt.bar(x + width/2, enhanced_times, width, label='Enhanced', alpha=0.7, color='green')
        plt.xlabel('Test Cases')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.xticks(x, test_ids, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        improvements = [(s - 1) * 100 for s in speedups]
        plt.pie(improvements, labels=test_ids, autopct='%1.1f%%', startangle=90)
        plt.title('Performance Improvement Distribution')
        
        plt.subplot(2, 2, 4)
        plt.scatter(original_times, enhanced_times, alpha=0.7, s=60)
        max_time = max(max(original_times), max(enhanced_times))
        plt.plot([0, max_time], [0, max_time], 'r--', alpha=0.7, label='No improvement')
        plt.xlabel('Original Time (seconds)')
        plt.ylabel('Enhanced Time (seconds)')
        plt.title('Time Correlation: Original vs Enhanced')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to: {output_path / 'performance_comparison.png'}")
    
    def _generate_detailed_report(self, results: Dict, output_path: Path):
        """Generate detailed markdown report."""
        report_content = []
        
        report_content.append("# Fire Simulation Performance Benchmark Report\n")
        report_content.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary section
        report_content.append("## Executive Summary\n")
        
        valid_speedups = [comp["speedup"] for comp in results["comparison"].values() 
                         if comp["speedup"] is not None]
        
        if valid_speedups:
            avg_speedup = np.mean(valid_speedups)
            report_content.append(f"- **Average Performance Improvement:** {avg_speedup:.2f}x speedup ({(avg_speedup-1)*100:.1f}% faster)\n")
            report_content.append(f"- **Best Case Improvement:** {np.max(valid_speedups):.2f}x speedup\n")
            report_content.append(f"- **Worst Case Improvement:** {np.min(valid_speedups):.2f}x speedup\n")
            
            total_time_saved = sum(comp["time_saved"] for comp in results["comparison"].values() 
                                 if comp["time_saved"] is not None)
            report_content.append(f"- **Total Time Saved:** {total_time_saved:.1f} seconds\n")
        
        report_content.append("\n## Detailed Results\n\n")
        
        # Detailed results table
        report_content.append("| Test Case | Original Time (s) | Enhanced Time (s) | Speedup | Improvement (%) |\n")
        report_content.append("|-----------|-------------------|-------------------|---------|----------------|\n")
        
        for test_id in results["comparison"].keys():
            orig_time = results["original"][test_id]["mean"]
            enh_time = results["enhanced"][test_id]["mean"]
            comparison = results["comparison"][test_id]
            
            if comparison["speedup"] is not None:
                report_content.append(
                    f"| {test_id} | {orig_time:.2f} | {enh_time:.2f} | "
                    f"{comparison['speedup']:.2f}x | {comparison['improvement_percent']:.1f}% |\n"
                )
            else:
                report_content.append(f"| {test_id} | {orig_time or 'Failed'} | {enh_time or 'Failed'} | N/A | N/A |\n")
        
        # Key improvements section
        report_content.append("\n## Key Improvements\n\n")
        report_content.append("The enhanced implementation includes the following improvements:\n\n")
        report_content.append("1. **Parallel Processing**: N=100 simulations now run in parallel using multiprocessing\n")
        report_content.append("2. **Better Memory Management**: Optimized data structures and reduced memory copying\n")
        report_content.append("3. **Improved Error Handling**: Robust error handling and logging\n")
        report_content.append("4. **Code Quality**: Type hints, better documentation, and modular design\n")
        report_content.append("5. **Configuration Management**: Centralized configuration and better defaults\n")
        
        # Write report
        with open(output_path / "performance_report.md", "w") as f:
            f.writelines(report_content)
        
        print(f"Detailed report saved to: {output_path / 'performance_report.md'}")


def main():
    """Main entry point for the benchmarking tool."""
    parser = argparse.ArgumentParser(description="Benchmark fire simulation performance")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per test case")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with smaller test cases")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(args.project_root)
    
    try:
        benchmark.setup_temp_environment()
        
        # Define test cases
        if args.quick:
            test_cases = [
                ("50x50", "benchmark_quick", "random", 10),
                ("50x50", "benchmark_quick", "degree", 15),
            ]
        else:
            test_cases = [
                ("100x100", "benchmark_small", "random", 10),
                ("100x100", "benchmark_small", "degree", 15),
                ("250x250", "benchmark_medium", "domirank", 20),
                ("250x250", "benchmark_medium", "bonacich", 25),
            ]
        
        # Run benchmark suite
        results = benchmark.run_benchmark_suite(test_cases, args.num_runs)
        
        # Generate report
        benchmark.generate_performance_report(results, args.output_dir)
        
        print("\nBenchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        sys.exit(1)
    finally:
        benchmark.cleanup_temp_environment()


if __name__ == "__main__":
    main()