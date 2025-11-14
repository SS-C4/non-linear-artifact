#!/usr/bin/env python3
"""
Run all experiments for different functions and collect timing data.
This script generates witnesses for all experiments and runs the prover,
collecting timing data into a CSV file.

Usage:
  python3 run_all.py                                    # Run all experiments
  python3 run_all.py --function <func>                  # Run all experiments for one function
  python3 run_all.py --function <func> --experiment <exp>  # Run one specific combination
  python3 run_all.py --list                             # List available functions and experiments
"""

import subprocess
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime

# Common parameters
NUM_INPUTS = 100
LOG_SCALES = [16, 32, 64, 120]

# Function configurations
# Each function is a tuple: (function_name, experiment_type, params_dict)
# params_dict maps log_scale -> list of params for that log_scale
EXPERIMENTS = {
    "inv_exp": {
        "poly": {
            16: ["inv_exp", NUM_INPUTS, 5, 16],  # Optimal
            32: ["inv_exp", NUM_INPUTS, 13, 32],  # Optimal (was 10)
            64: ["inv_exp", NUM_INPUTS, 21, 64],  # Optimal (was 18)
            120: ["inv_exp", NUM_INPUTS, 33, 120],  # Optimal (was 31)
        },
        "pade": {
            16: ["inv_exp", NUM_INPUTS, 3, 16],  # Optimal
            32: ["inv_exp", NUM_INPUTS, 5, 32],  # Optimal
            64: ["inv_exp", NUM_INPUTS, 8, 64],  # Optimal
            120: ["inv_exp", NUM_INPUTS, 16, 120],  # Fixed - was 13 (failed)
        },
        "pwl": {
            16: ["inv_exp", NUM_INPUTS, 0.0, 1.0, 5, 16],  # Optimal
            32: ["inv_exp", NUM_INPUTS, 0.0, 1.0, 13, 32],  # Optimal
            64: ["inv_exp", NUM_INPUTS, 0.0, 1.0, 13, 64],  # No optimal found, keeping original
            120: ["inv_exp", NUM_INPUTS, 0.0, 1.0, 13, 120],  # No optimal found, keeping original
        },
        "gl_quad": {
            16: ["inv_exp", NUM_INPUTS, 4, 16],  # Optimal
            32: ["inv_exp", NUM_INPUTS, 8, 32],  # Optimal
            64: ["inv_exp", NUM_INPUTS, 16, 64],  # Optimal
            120: ["inv_exp", NUM_INPUTS, 30, 120],  # Optimal
        },
        "lookup": {
            16: ["inv_exp", NUM_INPUTS, 8, 16],
            32: ["inv_exp", NUM_INPUTS, 8, 32],
            64: ["inv_exp", NUM_INPUTS, 8, 64],
            120: ["inv_exp", NUM_INPUTS, 8, 120],
        },
    },
    "sigmoid": {
        "poly": {
            16: ["sigmoid", NUM_INPUTS, 5, 16],  # Optimal (was 3)
            32: ["sigmoid", NUM_INPUTS, 13, 32],  # Optimal
            64: ["sigmoid", NUM_INPUTS, 33, 64],  # Optimal
            120: ["sigmoid", NUM_INPUTS, 69, 120],  # Optimal (was 65)
        },
        "pade": {
            16: ["sigmoid", NUM_INPUTS, 3, 16],  # Optimal
            32: ["sigmoid", NUM_INPUTS, 5, 32],  # Optimal
            64: ["sigmoid", NUM_INPUTS, 8, 64],  # Optimal
            120: ["sigmoid", NUM_INPUTS, 13, 120],  # Optimal
        },
        "pwl": {
            16: ["sigmoid", NUM_INPUTS, 0.0, 1.0, 3, 16],  # Optimal
            32: ["sigmoid", NUM_INPUTS, 0.0, 1.0, 11, 32],  # Optimal
            64: ["sigmoid", NUM_INPUTS, 0.0, 1.0, 13, 64],  # No optimal found, keeping original
            120: ["sigmoid", NUM_INPUTS, 0.0, 1.0, 13, 120],  # No optimal found, keeping original
        },
        "gl_quad": {
            16: ["sigmoid", NUM_INPUTS, 4, 16],  # Optimal
            32: ["sigmoid", NUM_INPUTS, 8, 32],  # Optimal
            64: ["sigmoid", NUM_INPUTS, 16, 64],  # Optimal
            120: ["sigmoid", NUM_INPUTS, 29, 120],  # Optimal
        },
        "lookup": {
            16: ["sigmoid", NUM_INPUTS, 8, 16],
            32: ["sigmoid", NUM_INPUTS, 8, 32],
            64: ["sigmoid", NUM_INPUTS, 8, 64],
            120: ["sigmoid", NUM_INPUTS, 8, 120],
        },
    },
    "gelu": {
        "poly": {
            16: ["gelu", NUM_INPUTS, 9, 16],  # Optimal (was 6)
            32: ["gelu", NUM_INPUTS, 17, 32],  # Optimal (was 16)
            64: ["gelu", NUM_INPUTS, 41, 64],  # Optimal (was 38)
            120: ["gelu", NUM_INPUTS, 77, 120],  # Optimal (was 76)
        },
        "pade": {
            16: ["gelu", NUM_INPUTS, 5, 16],  # Optimal
            32: ["gelu", NUM_INPUTS, 8, 32],  # Optimal
            64: ["gelu", NUM_INPUTS, 13, 64],  # Optimal (was 15)
            120: ["gelu", NUM_INPUTS, 24, 120],  # Optimal (was 25)
        },
        "pwl": {
            16: ["gelu", NUM_INPUTS, 0.0, 1.0, 5, 16],  # Optimal
            32: ["gelu", NUM_INPUTS, 0.0, 1.0, 13, 32],  # Optimal
            64: ["gelu", NUM_INPUTS, 0.0, 1.0, 13, 64],  # No optimal found, keeping original
            120: ["gelu", NUM_INPUTS, 0.0, 1.0, 13, 120],  # No optimal found, keeping original
        },
        "gl_quad": {
            16: ["gelu", NUM_INPUTS, 6, 16],  # Optimal (was 5)
            32: ["gelu", NUM_INPUTS, 12, 32],  # Optimal (was 11)
            64: ["gelu", NUM_INPUTS, 24, 64],  # Fixed - was 23 (failed)
            120: ["gelu", NUM_INPUTS, 44, 120],  # Optimal (was 43)
        },
        "lookup": {
            16: ["gelu", NUM_INPUTS, 8, 16],
            32: ["gelu", NUM_INPUTS, 8, 32],
            64: ["gelu", NUM_INPUTS, 8, 64],
            120: ["gelu", NUM_INPUTS, 8, 120],
        },
    },
    "erf": {
        "poly": {
            16: ["erf", NUM_INPUTS, 9, 16],  # Optimal
            32: ["erf", NUM_INPUTS, 21, 32],  # Optimal (was 19)
            64: ["erf", NUM_INPUTS, 37, 64],  # Optimal (was 35)
            120: ["erf", NUM_INPUTS, 61, 120],  # Optimal (was 59)
        },
        "pade": {
            16: ["erf", NUM_INPUTS, 5, 16],  # Optimal
            32: ["erf", NUM_INPUTS, 10, 32],  # Optimal
            64: ["erf", NUM_INPUTS, 17, 64],  # Optimal
            120: ["erf", NUM_INPUTS, 26, 120],  # Optimal
        },
        "pwl": {
            16: ["erf", NUM_INPUTS, 0.0, 1.0, 5, 16],  # Optimal
            32: ["erf", NUM_INPUTS, 0.0, 1.0, 13, 32],  # Optimal
            64: ["erf", NUM_INPUTS, 0.0, 1.0, 13, 64],  # No optimal found, keeping original
            120: ["erf", NUM_INPUTS, 0.0, 1.0, 13, 120],  # No optimal found, keeping original
        },
        "gl_quad": {
            16: ["erf", NUM_INPUTS, 4, 16],  # Fixed - was 3 (failed)
            32: ["erf", NUM_INPUTS, 8, 32],  # Optimal
            64: ["erf", NUM_INPUTS, 16, 64],  # Fixed - was 15 (failed)
            120: ["erf", NUM_INPUTS, 30, 120],  # Fixed - was 28 (failed)
        },
        "lookup": {
            16: ["erf", NUM_INPUTS, 16, 16],
            32: ["erf", NUM_INPUTS, 16, 32],
            64: ["erf", NUM_INPUTS, 16, 64],
            120: ["erf", NUM_INPUTS, 16, 120],
        },
    },
    "tanh": {
        "poly": {
            16: ["tanh", NUM_INPUTS, 13, 16],  # Optimal (was 16)
            32: ["tanh", NUM_INPUTS, 37, 32],  # Optimal (was 41)
            64: ["tanh", NUM_INPUTS, 85, 64],  # Optimal
            120: ["tanh", NUM_INPUTS, 150, 120],  # No optimal found, keeping original
        },
        "pade": {
            16: ["tanh", NUM_INPUTS, 4, 16],  # Optimal
            32: ["tanh", NUM_INPUTS, 6, 32],  # Optimal (was 8)
            64: ["tanh", NUM_INPUTS, 10, 64],  # Optimal
            120: ["tanh", NUM_INPUTS, 16, 120],  # Optimal
        },
        "pwl": {
            16: ["tanh", NUM_INPUTS, 0.0, 1.0, 5, 16],  # Optimal
            32: ["tanh", NUM_INPUTS, 0.0, 1.0, 13, 32],  # Optimal
            64: ["tanh", NUM_INPUTS, 0.0, 1.0, 13, 64],  # No optimal found, keeping original
            120: ["tanh", NUM_INPUTS, 0.0, 1.0, 13, 120],  # No optimal found, keeping original
        },
        "gl_quad": {
            16: ["tanh", NUM_INPUTS, 7, 16],  # Optimal
            32: ["tanh", NUM_INPUTS, 14, 32],  # Optimal
            64: ["tanh", NUM_INPUTS, 28, 64],  # Optimal
            120: ["tanh", NUM_INPUTS, 56, 120],  # Fixed - was 43 (failed)
        },
        "lookup": {
            16: ["tanh", NUM_INPUTS, 8, 16],
            32: ["tanh", NUM_INPUTS, 8, 32],
            64: ["tanh", NUM_INPUTS, 8, 64],
            120: ["tanh", NUM_INPUTS, 8, 120],
        },
    },
    "tan": {
        "poly": {
            16: ["tan", NUM_INPUTS, 17, 16],  # Optimal (was 16)
            32: ["tan", NUM_INPUTS, 37, 32],  # Optimal (was 38)
            64: ["tan", NUM_INPUTS, 90, 64],  # Fixed - was 81 (failed)
            120: ["tan", NUM_INPUTS, 80, 120],  # No optimal found, keeping original (OK to fail)
        },
        "pade": {
            16: ["tan", NUM_INPUTS, 4, 16],  # Optimal
            32: ["tan", NUM_INPUTS, 7, 32],  # Optimal (was 8)
            64: ["tan", NUM_INPUTS, 11, 64],  # Optimal (was 10)
            120: ["tan", NUM_INPUTS, 16, 120],  # Optimal
        },
        "pwl": {
            16: ["tan", NUM_INPUTS, 0.0, 0.78, 7, 16],  # Optimal (was 5)
            32: ["tan", NUM_INPUTS, 0.0, 0.78, 14, 32],  # No optimal found, keeping original
            64: ["tan", NUM_INPUTS, 0.0, 0.78, 14, 64],  # No optimal found, keeping original
            120: ["tan", NUM_INPUTS, 0.0, 0.78, 14, 120],  # No optimal found, keeping original
        },
        "gl_quad": {
            16: ["tan", NUM_INPUTS, 4, 16],  # Optimal (was 5)
            32: ["tan", NUM_INPUTS, 9, 32],  # Optimal (was 10)
            64: ["tan", NUM_INPUTS, 18, 64],  # Optimal
            120: ["tan", NUM_INPUTS, 34, 120],  # Optimal
        },
        "lookup": {
            16: ["tan", NUM_INPUTS, 16, 16],  
            32: ["tan", NUM_INPUTS, 16, 32],
            64: ["tan", NUM_INPUTS, 16, 64],
            120: ["tan", NUM_INPUTS, 16, 120],
        },
    },
    "cos": {
        "poly": {
            16: ["cos", NUM_INPUTS, 5, 16],  # Optimal (was 6)
            32: ["cos", NUM_INPUTS, 13, 32],  # Optimal (was 11)
            64: ["cos", NUM_INPUTS, 21, 64],  # Optimal (was 19)
            120: ["cos", NUM_INPUTS, 33, 120],  # Optimal (was 32)
        },
        "pade": {
            16: ["cos", NUM_INPUTS, 4, 16],  # Optimal
            32: ["cos", NUM_INPUTS, 6, 32],  # Optimal (was 10)
            64: ["cos", NUM_INPUTS, 10, 64],  # Optimal
            120: ["cos", NUM_INPUTS, 16, 120],  # Optimal
        },
        "pwl": {
            16: ["cos", NUM_INPUTS, 0.0, 1.0, 5, 16],  # Optimal (was 6)
            32: ["cos", NUM_INPUTS, 0.0, 1.0, 13, 32],  # Optimal (was 14)
            64: ["cos", NUM_INPUTS, 0.0, 1.0, 14, 64],  # No optimal found, keeping original
            120: ["cos", NUM_INPUTS, 0.0, 1.0, 14, 120],  # No optimal found, keeping original
        },
        "gl_quad": {
            16: ["cos", NUM_INPUTS, 4, 16],  # Optimal (was 5)
            32: ["cos", NUM_INPUTS, 9, 32],  # Optimal (was 10)
            64: ["cos", NUM_INPUTS, 18, 64],  # Optimal
            120: ["cos", NUM_INPUTS, 34, 120],  # Optimal
        },
        "lookup": {
            16: ["cos", NUM_INPUTS, 16, 16],  # log_base = log_scale for lookup tables
            32: ["cos", NUM_INPUTS, 16, 32],
            64: ["cos", NUM_INPUTS, 16, 64],
            120: ["cos", NUM_INPUTS, 16, 120],
        },
    },
    "power": {
        "poly": {
            16: ["power", NUM_INPUTS, 5, 16, 0.876],  # Optimal (was 6, 0.745) - using 0.876 from optimal
            32: ["power", NUM_INPUTS, 9, 32, 0.876],  # Optimal (was 11, 0.745)
            64: ["power", NUM_INPUTS, 17, 64, 0.876],  # Optimal (was 19, 0.745)
            120: ["power", NUM_INPUTS, 29, 120, 0.876],  # Optimal (was 27, 0.745)
        },
        "pade": {
            16: ["power", NUM_INPUTS, 3, 16, 0.876],  # Optimal (was 4)
            32: ["power", NUM_INPUTS, 5, 32, 0.876],  # Optimal (was 7)
            64: ["power", NUM_INPUTS, 8, 64, 0.876],  # Optimal (was 10)
            120: ["power", NUM_INPUTS, 13, 120, 0.876],  # Optimal
        },
        "pwl": {
            16: ["power", NUM_INPUTS, 0.0, 1.0, 7, 16, 0.876],  # No optimal found, keeping original
            32: ["power", NUM_INPUTS, 0.0, 1.0, 14, 32, 0.876],  # No optimal found, keeping original
            64: ["power", NUM_INPUTS, 0.0, 1.0, 14, 64, 0.876],  # No optimal found, keeping original
            120: ["power", NUM_INPUTS, 0.0, 1.0, 14, 120, 0.876],  # No optimal found, keeping original
        },
        "gl_quad": {
            16: ["power", NUM_INPUTS, 3, 16, 0.876],  # Optimal
            32: ["power", NUM_INPUTS, 6, 32, 0.876],  # Optimal
            64: ["power", NUM_INPUTS, 13, 64, 0.876],  # Fixed - was 12 (failed)
            120: ["power", NUM_INPUTS, 23, 120, 0.876],  # Optimal (was 26)
        },
        "lookup": {
            16: ["power", NUM_INPUTS, 8, 16, 0.876],
            32: ["power", NUM_INPUTS, 8, 32, 0.876],
            64: ["power", NUM_INPUTS, 8, 64, 0.876],
            120: ["power", NUM_INPUTS, 8, 120, 0.876],
        },
    },
}

# Script mapping
GENERATOR_SCRIPTS = {
    "poly": "poly_gen.py",
    "pade": "pade_gen.py",
    "pwl": "pwl_gen.py",
    "gl_quad": "gl_gen.py",
    "lookup": "lookup_gen.py",
}

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"✗ Failed with exit code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("✓ Success")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"✗ Timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def generate_witness(experiment_type, params):
    """Generate witness using the appropriate generator script."""
    script = GENERATOR_SCRIPTS[experiment_type]
    cmd = ["python3", f"scripts/{script}"] + [str(p) for p in params]
    
    return run_command(
        cmd,
        f"Generating witness for {experiment_type}: {params[0]}"
    )

def run_prover(experiment_type):
    """Run the prover and extract timing data."""
    print(f"\n{'='*60}")
    print(f"Running prover for {experiment_type}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            ["python3", "scripts/run_prover.py", experiment_type, "--quiet"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for proving
        )
        
        if result.returncode != 0:
            print(f"✗ Prover failed with exit code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        # Parse output: pk_time,prover_time,verifier_time,proof_size
        output_lines = result.stdout.strip().split('\n')
        # Get the last line which should contain the data
        data_line = output_lines[-1]
        
        parts = data_line.split(',')
        if len(parts) >= 3:
            pk_time = float(parts[0]) if parts[0] != 'None' else None
            prover_time = float(parts[1]) if parts[1] != 'None' else None
            verifier_time = float(parts[2]) if parts[2] != 'None' else None
            proof_size = parts[3].strip() if len(parts) > 3 else None
            
            print(f"✓ Prover completed successfully")
            print(f"  PK Time: {pk_time} ms")
            print(f"  Prover Time: {prover_time} ms")
            print(f"  Verifier Time: {verifier_time} ms")
            print(f"  Proof Size: {proof_size}")
            
            return {
                "pk_time_ms": pk_time,
                "prover_time_ms": prover_time,
                "verifier_time_ms": verifier_time,
                "proof_size": proof_size
            }
        else:
            print(f"✗ Could not parse prover output: {data_line}")
            return None
        
    except subprocess.TimeoutExpired:
        print(f"✗ Prover timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"✗ Error running prover: {e}")
        return None

def list_available():
    """List all available functions and experiments."""
    print("\nAvailable Functions:")
    for func_name in EXPERIMENTS.keys():
        print(f"  - {func_name}")
    
    print("\nAvailable Experiment Types:")
    for exp_type in GENERATOR_SCRIPTS.keys():
        print(f"  - {exp_type}")
    
    print("\nExample Usage:")
    print("  python3 run_all.py                                    # Run all experiments")
    print("  python3 run_all.py --function sigmoid                 # Run all sigmoid experiments")
    print("  python3 run_all.py --function sigmoid --experiment poly  # Run sigmoid polynomial only")
    print()

def main():
    """Main function to run all experiments."""
    parser = argparse.ArgumentParser(
        description="Run non-linear approximation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--function",
        type=str,
        help="Run experiments for a specific function (e.g., sigmoid, gelu, tanh)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Run a specific experiment type (e.g., poly, pade, pwl, gl_quad, lookup)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available functions and experiments"
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_available()
        return
    
    # Validate arguments
    if args.function and args.function not in EXPERIMENTS:
        print(f"Error: Unknown function '{args.function}'")
        print(f"Available functions: {', '.join(EXPERIMENTS.keys())}")
        sys.exit(1)
    
    if args.experiment and args.experiment not in GENERATOR_SCRIPTS:
        print(f"Error: Unknown experiment type '{args.experiment}'")
        print(f"Available experiments: {', '.join(GENERATOR_SCRIPTS.keys())}")
        sys.exit(1)
    
    # Filter experiments based on arguments
    if args.function:
        experiments_to_run = {args.function: EXPERIMENTS[args.function]}
        run_description = f"Function: {args.function.upper()}"
        if args.experiment:
            experiments_to_run = {
                args.function: {
                    args.experiment: EXPERIMENTS[args.function][args.experiment]
                }
            }
            run_description = f"Function: {args.function.upper()}, Experiment: {args.experiment}"
    else:
        experiments_to_run = EXPERIMENTS
        run_description = "All Experiments"
    
    print(f"\n{'#'*80}")
    print(f"# Running: {run_description}")
    print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# NUM_INPUTS: {NUM_INPUTS}")
    print(f"# LOG_SCALES: {LOG_SCALES}")
    print(f"{'#'*80}\n")
    
    # Create experiments directory if it doesn't exist
    experiments_dir = Path(__file__).parent.parent / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    
    # Prepare CSV file with descriptive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.function and args.experiment:
        csv_filename = f"results_{args.function}_{args.experiment}_{timestamp}.csv"
    elif args.function:
        csv_filename = f"results_{args.function}_{timestamp}.csv"
    else:
        csv_filename = f"results_all_{timestamp}.csv"
    
    csv_path = experiments_dir / csv_filename
    
    results = []
    
    # Run experiments
    for func_name, func_experiments in experiments_to_run.items():
        print(f"\n{'#'*80}")
        print(f"# Function: {func_name.upper()}")
        print(f"{'#'*80}\n")
        
        for experiment_type, scale_params in func_experiments.items():
            for log_scale in LOG_SCALES:
                if log_scale not in scale_params:
                    print(f"⚠ Skipping {func_name}/{experiment_type}/log_scale={log_scale} (not configured)")
                    continue
                
                params = scale_params[log_scale]
                
                print(f"\n{'*'*60}")
                print(f"* Running: {func_name} / {experiment_type} / log_scale={log_scale}")
                print(f"* Parameters: {params}")
                print(f"{'*'*60}")
                
                # Generate witness
                if not generate_witness(experiment_type, params):
                    print(f"✗ Skipping prover due to witness generation failure")
                    results.append({
                        "function": func_name,
                        "experiment": experiment_type,
                        "log_scale": log_scale,
                        "num_inputs": NUM_INPUTS,
                        "params": str(params),
                        "pk_time_ms": "FAILED",
                        "prover_time_ms": "FAILED",
                        "verifier_time_ms": "FAILED",
                        "proof_size": "FAILED"
                    })
                    continue
                
                # Run prover
                timing_data = run_prover(experiment_type)
                
                if timing_data is None:
                    print(f"✗ Prover failed")
                    results.append({
                        "function": func_name,
                        "experiment": experiment_type,
                        "log_scale": log_scale,
                        "num_inputs": NUM_INPUTS,
                        "params": str(params),
                        "pk_time_ms": "FAILED",
                        "prover_time_ms": "FAILED",
                        "verifier_time_ms": "FAILED",
                        "proof_size": "FAILED"
                    })
                else:
                    results.append({
                        "function": func_name,
                        "experiment": experiment_type,
                        "log_scale": log_scale,
                        "num_inputs": NUM_INPUTS,
                        "params": str(params),
                        **timing_data
                    })
                
                # Write results incrementally to CSV
                with open(csv_path, 'w', newline='') as csvfile:
                    if results:
                        fieldnames = list(results[0].keys())
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)
                
                print(f"\n✓ Results saved to {csv_path}")
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# Experiments Completed")
    print(f"# Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total experiments run: {len(results)}")
    print(f"# Results saved to: {csv_path}")
    print(f"{'#'*80}\n")
    
    # Print summary statistics
    successful = sum(1 for r in results if r["pk_time_ms"] != "FAILED")
    failed = len(results) - successful
    
    print(f"Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed:     {failed}")
    
    if successful > 0:
        print(f"\n✓ Experiments completed. Results saved to: {csv_path}")
        print(f"  CSV file: {csv_path.relative_to(Path(__file__).parent.parent)}")
    else:
        print(f"\n✗ All experiments failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
