#!/bin/bash
# Enhanced Fire Simulation Pipeline
# Improvements:
# - Better error handling and logging
# - Configuration-driven approach
# - Uses enhanced parallel simulation script
# - Cleaner code structure and documentation

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
handle_error() {
    log_error "Script failed at line $1. Exiting..."
    exit 1
}

trap 'handle_error $LINENO' ERR

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# Load environment
if [[ -f "$PROJECT_ROOT/env.sh" ]]; then
    source "$PROJECT_ROOT/env.sh"
    log_info "Loaded environment configuration"
else
    log_warning "env.sh not found, proceeding with default environment"
fi

# Simulation configuration
readonly XLEN=${GRID_WIDTH:-250}
readonly YLEN=${GRID_HEIGHT:-250}
readonly SAVENAME=${SIMULATION_NAME:-"mean_full_250_4neighbors_enhanced"}
readonly CENTRALITIES=(domirank random degree bonacich)
readonly PERCENTAGES=(0 5 10 15 20 25 30)

# Performance configuration
readonly NUM_CORES=$(nproc)
readonly MAX_JOBS=${MAX_PARALLEL_JOBS:-$((NUM_CORES > 1 ? NUM_CORES - 1 : 1))}

log_info "Starting enhanced fire simulation pipeline"
log_info "Configuration:"
log_info "  Grid size: ${XLEN}x${YLEN}"
log_info "  Save name: $SAVENAME"
log_info "  Max parallel jobs: $MAX_JOBS"
log_info "  CPU cores available: $NUM_CORES"

# Helper function to manage parallel jobs
throttle_jobs() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        sleep 0.5
    done
}

# Validation function
validate_dependencies() {
    local missing_deps=()
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check for required Python scripts
    local required_scripts=(
        "src/scripts/create_adjacency.py"
        "src/scripts/generate_fuel_breaks.py"
        "src/scripts/simulate_average_enhanced.py"
        "src/scripts/generate_plots.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$script" ]]; then
            missing_deps+=("$script")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies:"
        printf '  %s\n' "${missing_deps[@]}"
        exit 1
    fi
    
    log_success "All dependencies validated"
}

# Create directory structure
setup_directories() {
    local results_dir="$PROJECT_ROOT/src/results"
    mkdir -p "$results_dir"
    log_info "Created results directory: $results_dir"
}

# Step 1: Create adjacency matrix
create_adjacency() {
    log_info "Creating adjacency matrix..."
    
    if python3 src/scripts/create_adjacency.py "${XLEN}x${YLEN}" "$SAVENAME"; then
        log_success "Adjacency matrix created successfully"
    else
        log_error "Failed to create adjacency matrix"
        return 1
    fi
}

# Step 2: Generate fuel breaks for all centralities
generate_fuel_breaks() {
    log_info "Generating fuel breaks for all centralities..."
    
    local failed_centralities=()
    
    for centrality in "${CENTRALITIES[@]}"; do
        log_info "  Processing centrality: $centrality"
        
        if python3 src/scripts/generate_fuel_breaks.py \
            "${XLEN}x${YLEN}" \
            "$SAVENAME" \
            "$centrality"; then
            log_success "  Fuel breaks generated for $centrality"
        else
            log_error "  Failed to generate fuel breaks for $centrality"
            failed_centralities+=("$centrality")
        fi
    done
    
    if [[ ${#failed_centralities[@]} -gt 0 ]]; then
        log_warning "Failed centralities: ${failed_centralities[*]}"
        log_warning "Continuing with successful centralities..."
    fi
    
    log_success "Fuel break generation completed"
}

# Step 3: Run parallel simulations (using enhanced script)
run_simulations() {
    log_info "Starting parallel fire simulations..."
    log_info "This will run ${#PERCENTAGES[@]} × ${#CENTRALITIES[@]} = $((${#PERCENTAGES[@]} * ${#CENTRALITIES[@]})) simulations"
    
    local total_jobs=0
    local completed_jobs=0
    local failed_jobs=0
    local job_pids=()
    
    # Create a temporary directory for job tracking
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    for percentage in "${PERCENTAGES[@]}"; do
        for centrality in "${CENTRALITIES[@]}"; do
            throttle_jobs
            
            (
                local job_id="${centrality}_${percentage}"
                local start_time=$(date +%s)
                local status_file="$temp_dir/${job_id}.status"
                
                log_info "Starting simulation: $centrality @ ${percentage}% fuel breaks"
                
                if python3 src/scripts/simulate_average_enhanced.py \
                    "${XLEN}x${YLEN}" \
                    "$SAVENAME" \
                    "$centrality" \
                    "$percentage" > "$temp_dir/${job_id}.log" 2>&1; then
                    
                    local end_time=$(date +%s)
                    local duration=$((end_time - start_time))
                    echo "SUCCESS $duration" > "$status_file"
                    log_success "Completed simulation: $job_id (${duration}s)"
                else
                    echo "FAILED" > "$status_file"
                    log_error "Failed simulation: $job_id"
                    log_error "Check log: $temp_dir/${job_id}.log"
                fi
            ) &
            
            job_pids+=($!)
            ((total_jobs++))
        done
    done
    
    # Wait for all jobs and collect results
    log_info "Waiting for $total_jobs simulation jobs to complete..."
    
    for pid in "${job_pids[@]}"; do
        wait "$pid" || ((failed_jobs++))
        ((completed_jobs++))
        
        if ((completed_jobs % 5 == 0)) || ((completed_jobs == total_jobs)); then
            log_info "Progress: $completed_jobs/$total_jobs jobs completed"
        fi
    done
    
    # Report results
    local successful_jobs=$((total_jobs - failed_jobs))
    log_info "Simulation results:"
    log_info "  Total jobs: $total_jobs"
    log_info "  Successful: $successful_jobs"
    log_info "  Failed: $failed_jobs"
    
    if [[ $failed_jobs -gt 0 ]]; then
        log_warning "Some simulations failed. Check logs in: $temp_dir"
        # Copy failed logs to results directory for debugging
        local results_dir="src/results/$SAVENAME"
        mkdir -p "$results_dir/failed_logs"
        cp "$temp_dir"/*.log "$results_dir/failed_logs/" 2>/dev/null || true
    fi
    
    if [[ $successful_jobs -eq 0 ]]; then
        log_error "All simulations failed!"
        return 1
    fi
    
    log_success "Parallel simulations completed successfully"
}

# Step 4: Generate plots
generate_plots() {
    log_info "Generating plots..."
    
    if python3 src/scripts/generate_plots.py \
        "${XLEN}x${YLEN}" \
        "$SAVENAME"; then
        log_success "Plots generated successfully"
    else
        log_error "Failed to generate plots"
        return 1
    fi
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    log_info "=================================================="
    log_info "Enhanced Fire Simulation Pipeline"
    log_info "=================================================="
    
    # Validation and setup
    validate_dependencies
    setup_directories
    
    # Execute pipeline steps
    create_adjacency
    generate_fuel_breaks
    run_simulations
    generate_plots
    
    # Final summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))
    local seconds=$((total_duration % 60))
    
    log_success "=================================================="
    log_success "Pipeline completed successfully!"
    log_success "Total runtime: ${hours}h ${minutes}m ${seconds}s"
    log_success "Results saved in: src/results/$SAVENAME"
    log_success "=================================================="
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi