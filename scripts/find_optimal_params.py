#!/usr/bin/env python3
"""
Find optimal (smallest) parameters for each experiment type and function.
This script runs experiments with increasing parameter values until the first success,
then records the optimal parameters.

Usage:
  python3 scripts/find_optimal_params.py                    # Run all functions from the start
  python3 scripts/find_optimal_params.py --start-from gelu  # Resume from 'gelu' function
  python3 scripts/find_optimal_params.py --load-baseline experiments/optimal_params_20251112_010020.txt
  python3 scripts/find_optimal_params.py --function sigmoid                 # Run only sigmoid
  python3 scripts/find_optimal_params.py --function sigmoid --experiment poly  # Run sigmoid poly only
  python3 scripts/find_optimal_params.py --list             # List available functions and experiments
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Common parameters
NUM_INPUTS = 100
LOG_SCALES = [16, 32, 64, 120]

# Functions to test (excluding softmax as per original requirements)
FUNCTIONS = ["inv_exp", "sigmoid", "gelu", "erf", "tanh", "tan", "cos", "power"]

# Experiment configurations with search ranges
# Format: (start_value, max_value, step)
EXPERIMENT_CONFIGS = {
    "poly": {
        "param_name": "degree",
        "start": 1,
        "max": 100,
        "step": 4,
        "param_index": 2,  # position in command args
    },
    "pade": {
        "param_name": "degree",
        "start": 1,
        "max": 50,
        "step": 1,
        "param_index": 2,
    },
    "pwl": {
        "param_name": "log_num_pieces",
        "start": 1,
        "max": 14,  
        "step": 1,
        "param_index": 4,
    },
    "gl_quad": {
        "param_name": "n_points",
        "start": 1,
        "max": 50,
        "step": 1,
        "param_index": 2,
    },
}

# Generator script mapping
GENERATOR_SCRIPTS = {
    "poly": "poly_gen.py",
    "pade": "pade_gen.py",
    "pwl": "pwl_gen.py",
    "gl_quad": "gl_gen.py",
}

# Fixed parameters for each experiment type
def get_base_params(func_name, experiment_type, log_scale):
    """Get base parameters for each experiment type."""
    if experiment_type == "poly":
        if func_name == "power":
            return [func_name, NUM_INPUTS, None, log_scale, 0.876]  # None = degree to be tested
        else:
            return [func_name, NUM_INPUTS, None, log_scale]
    
    elif experiment_type == "pade":
        if func_name == "power":
            return [func_name, NUM_INPUTS, None, log_scale, 0.876]
        else:
            return [func_name, NUM_INPUTS, None, log_scale]
    
    elif experiment_type == "pwl":
        # Domain boundaries based on function
        if func_name == "power":
            return [func_name, NUM_INPUTS, 1.0, 2.0, None, log_scale, 0.876]
        elif func_name == "tan":
            return [func_name, NUM_INPUTS, 0.0, 1.0, None, log_scale]
        elif func_name == "cos":
            return [func_name, NUM_INPUTS, 0.0, 1.0, None, log_scale]
        else:
            return [func_name, NUM_INPUTS, 0.0, 1.0, None, log_scale]
    
    elif experiment_type == "gl_quad":
        if func_name == "power":
            return [func_name, NUM_INPUTS, None, log_scale, 0.876]
        else:
            return [func_name, NUM_INPUTS, None, log_scale]
    
    return None

def run_witness_generation(experiment_type, params):
    """Try to generate witness with given parameters."""
    script = GENERATOR_SCRIPTS[experiment_type]
    cmd = ["python3", f"scripts/{script}"] + [str(p) for p in params]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except:
        return False

def run_prover(experiment_type):
    """Run the prover to verify the circuit works."""
    try:
        result = subprocess.run(
            ["python3", "scripts/run_prover.py", experiment_type, "--execute-only", "--quiet"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except:
        return False

def find_optimal_param(func_name, experiment_type, log_scale, baseline_params=None):
    """Find the smallest parameter value that works.
    
    Args:
        func_name: Name of the function to test
        experiment_type: Type of experiment (poly, pade, pwl, gl_quad)
        log_scale: Log scale value
        baseline_params: Dict with baseline optimal values from first function (inv_exp)
                        Format: {(experiment_type, log_scale): optimal_value}
    """
    config = EXPERIMENT_CONFIGS[experiment_type]
    param_name = config["param_name"]
    param_index = config["param_index"]
    
    print(f"\n{'='*70}")
    print(f"Searching: {func_name} / {experiment_type} / log_scale={log_scale}")
    print(f"{'='*70}")
    
    # Get base parameters
    base_params = get_base_params(func_name, experiment_type, log_scale)
    if base_params is None:
        print(f"✗ No base parameters defined")
        return None
    
    # Find the None position (parameter to optimize)
    none_index = base_params.index(None)
    
    # Determine starting point
    start_value = config["start"]
    if baseline_params and (experiment_type, log_scale) in baseline_params:
        baseline_value = baseline_params[(experiment_type, log_scale)]
        # Start from baseline minus 4, but not less than config start
        start_value = max(config["start"], baseline_value - 4)
        print(f"  Using baseline from inv_exp: {baseline_value}, starting from {start_value}")
    
    # Search for optimal value
    for param_value in range(start_value, config["max"] + 1, config["step"]):
        test_params = base_params.copy()
        test_params[none_index] = param_value
        
        print(f"  Testing {param_name}={param_value}...", end=" ", flush=True)
        
        # Try witness generation
        if not run_witness_generation(experiment_type, test_params):
            print("✗ witness generation failed")
            continue
        
        # Try proving
        if not run_prover(experiment_type):
            print("✗ prover failed")
            continue
        
        # Success!
        print("✓ SUCCESS")
        result = {
            "function": func_name,
            "experiment": experiment_type,
            "log_scale": log_scale,
            "param_name": param_name,
            "param_value": param_value,
            "command": f"python3 scripts/{GENERATOR_SCRIPTS[experiment_type]} {' '.join(str(p) for p in test_params)}"
        }
        return result
    
    print(f"  ✗ No valid {param_name} found in range [{config['start']}, {config['max']}]")
    return None

def load_baseline_from_file(filepath):
    """Load baseline parameters from a previous results file.
    
    Args:
        filepath: Path to the optimal_params_*.txt file
        
    Returns:
        Dictionary mapping (experiment_type, log_scale) to optimal_value
    """
    baseline_params = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for lines with optimal parameter info
        if line.startswith("# ") and " / " in line and " / log_scale=" in line:
            # Parse: # inv_exp / poly / log_scale=16
            parts = line[2:].split(" / ")
            if len(parts) == 3:
                func_name = parts[0].strip()
                experiment_type = parts[1].strip()
                log_scale = int(parts[2].split("=")[1].strip())
                
                # Next line should have the optimal value
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("# Optimal"):
                        # Parse: # Optimal degree: 5
                        param_value = int(next_line.split(":")[-1].strip())
                        
                        # Only store if it's from inv_exp (the baseline function)
                        if func_name == "inv_exp":
                            baseline_params[(experiment_type, log_scale)] = param_value
                            print(f"  Loaded baseline: {experiment_type} / log_scale={log_scale} -> {param_value}")
        
        i += 1
    
    return baseline_params

def list_available():
    """List all available functions and experiments."""
    print("\nAvailable Functions:")
    for func_name in FUNCTIONS:
        print(f"  - {func_name}")
    
    print("\nAvailable Experiment Types:")
    for exp_type in EXPERIMENT_CONFIGS.keys():
        print(f"  - {exp_type}")
    
    print("\nExample Usage:")
    print("  python3 scripts/find_optimal_params.py                                # Run all")
    print("  python3 scripts/find_optimal_params.py --function sigmoid             # Run only sigmoid")
    print("  python3 scripts/find_optimal_params.py --function sigmoid --experiment poly  # Run sigmoid poly only")
    print("  python3 scripts/find_optimal_params.py --start-from gelu              # Resume from gelu")
    print()

def main():
    """Main function to find optimal parameters for all experiments."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Find optimal parameters for non-linear approximations")
    parser.add_argument("--start-from", type=str, default=None,
                        help=f"Start from a specific function. Options: {', '.join(FUNCTIONS)}")
    parser.add_argument("--load-baseline", type=str, default=None,
                        help="Load baseline parameters from a previous results file")
    parser.add_argument("--function", type=str, default=None,
                        help="Run optimization for a specific function only")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run optimization for a specific experiment type only")
    parser.add_argument("--list", action="store_true",
                        help="List available functions and experiments")
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_available()
        return
    
    # Validate arguments
    start_function = args.start_from
    if start_function and start_function not in FUNCTIONS:
        print(f"Error: Invalid function '{start_function}'")
        print(f"Available functions: {', '.join(FUNCTIONS)}")
        sys.exit(1)
    
    if args.function and args.function not in FUNCTIONS:
        print(f"Error: Unknown function '{args.function}'")
        print(f"Available functions: {', '.join(FUNCTIONS)}")
        sys.exit(1)
    
    if args.experiment and args.experiment not in EXPERIMENT_CONFIGS:
        print(f"Error: Unknown experiment type '{args.experiment}'")
        print(f"Available experiments: {', '.join(EXPERIMENT_CONFIGS.keys())}")
        sys.exit(1)
    
    # --function overrides --start-from
    if args.function and args.start_from:
        print("Warning: --function overrides --start-from, ignoring --start-from")
        start_function = None
    
    # Build run description
    run_description = "All Functions"
    if args.function and args.experiment:
        run_description = f"Function: {args.function.upper()}, Experiment: {args.experiment}"
    elif args.function:
        run_description = f"Function: {args.function.upper()}"
    elif start_function:
        run_description = f"Starting from: {start_function.upper()}"
    
    print(f"\n{'#'*80}")
    print(f"# Finding Optimal Parameters")
    print(f"# Running: {run_description}")
    print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# NUM_INPUTS: {NUM_INPUTS}")
    print(f"# LOG_SCALES: {LOG_SCALES}")
    if args.load_baseline:
        print(f"# Loading baseline from: {args.load_baseline}")
    print(f"{'#'*80}\n")
    
    # Create experiments directory
    experiments_dir = Path(__file__).parent.parent / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    
    # Output file for optimal parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = experiments_dir / f"optimal_params_{timestamp}.txt"
    
    results = []
    
    # Dictionary to store baseline optimal values from inv_exp
    # Format: {(experiment_type, log_scale): optimal_value}
    baseline_params = {}
    
    # Load baseline from file if specified
    if args.load_baseline:
        baseline_file = Path(args.load_baseline)
        if baseline_file.exists():
            print(f"\nLoading baseline parameters from {baseline_file}...")
            baseline_params = load_baseline_from_file(baseline_file)
            print(f"✓ Loaded {len(baseline_params)} baseline values\n")
        else:
            print(f"Warning: Baseline file not found: {baseline_file}")
            print("Continuing without baseline...\n")
    
    # Determine which functions to run
    if args.function:
        functions_to_run = [args.function]
        print(f"Running single function: {args.function}")
    elif start_function:
        start_idx = FUNCTIONS.index(start_function)
        functions_to_run = FUNCTIONS[start_idx:]
        print(f"Starting from function: {start_function} (index {start_idx})")
        print(f"Functions to run: {', '.join(functions_to_run)}")
    else:
        functions_to_run = FUNCTIONS
    
    # Determine which experiments to run
    if args.experiment:
        experiments_to_run = [args.experiment]
        print(f"Running single experiment type: {args.experiment}")
    else:
        experiments_to_run = list(EXPERIMENT_CONFIGS.keys())
    
    print(f"Functions: {', '.join(functions_to_run)}")
    print(f"Experiments: {', '.join(experiments_to_run)}")
    print()
    
    # Test each function, experiment type, and log scale
    for func_idx, func_name in enumerate(functions_to_run):
        print(f"\n{'#'*80}")
        print(f"# Function: {func_name.upper()}")
        print(f"{'#'*80}")
        
        # For the first function (inv_exp), pass None as baseline unless loaded from file
        # For subsequent functions, pass the baseline_params dictionary
        is_first_function = (func_name == "inv_exp")
        has_baseline = len(baseline_params) > 0
        
        for experiment_type in experiments_to_run:
            print(f"\n## Experiment: {experiment_type}")
            
            for log_scale in LOG_SCALES:
                if is_first_function and not has_baseline:
                    # First function (inv_exp) with no loaded baseline - search from start
                    result = find_optimal_param(func_name, experiment_type, log_scale, baseline_params=None)
                    
                    # Store the optimal value as baseline for subsequent functions
                    if result:
                        baseline_params[(experiment_type, log_scale)] = result['param_value']
                elif is_first_function and has_baseline:
                    # First function (inv_exp) but baseline was loaded - still use it and update
                    result = find_optimal_param(func_name, experiment_type, log_scale, baseline_params=baseline_params)
                    
                    # Update baseline with new result
                    if result:
                        baseline_params[(experiment_type, log_scale)] = result['param_value']
                else:
                    # Subsequent functions - use baseline from inv_exp
                    result = find_optimal_param(func_name, experiment_type, log_scale, baseline_params=baseline_params)
                
                if result:
                    results.append(result)
                    
                    # Write results incrementally
                    with open(output_file, 'w') as f:
                        f.write(f"# Optimal Parameters Found\n")
                        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"# NUM_INPUTS: {NUM_INPUTS}\n\n")
                        
                        for r in results:
                            f.write(f"# {r['function']} / {r['experiment']} / log_scale={r['log_scale']}\n")
                            f.write(f"# Optimal {r['param_name']}: {r['param_value']}\n")
                            f.write(f"{r['command']}\n\n")
                    
                    print(f"\n✓ Results saved to {output_file.relative_to(Path.cwd())}")
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# Optimization Complete")
    print(f"# Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total successful optimizations: {len(results)}")
    print(f"# Results saved to: {output_file}")
    print(f"{'#'*80}\n")
    
    if len(results) > 0:
        print(f"✓ Found optimal parameters for {len(results)} configurations")
    else:
        print(f"✗ No optimal parameters found")
        sys.exit(1)

if __name__ == "__main__":
    main()
