#!/usr/bin/env python3
import sys
import re
import subprocess
from pathlib import Path

# Map experiment names to their witness types
EXPERIMENT_MAP = {
    "lookup": "lookup_wits: [LookupWitness<LOOKUP_FUNC_TYPE>; LOOKUP_NUM]",
    "gl_quad": "gl_wits: [GLWitness<GL_FUNC_TYPE>; GL_NUM]",
    "pwl": "pwl_wits: [PWLWitness; PWL_NUM]",
    "poly": "poly_wits: [PolyWitness; POLY_NUM]",
    "pade": "pade_wits: [PadeWitness; PADE_NUM]",
}

def print_usage():
    print("Usage: python run_prover.py <experiment>")
    print("\nExperiments:")
    print("  lookup    - Lookup-based approximation")
    print("  gl_quad   - Gauss-Legendre quadrature")
    print("  pwl       - Piecewise linear approximation")
    print("  poly      - Polynomial approximation")
    print("  pade      - Padé approximation")
    print("\nExample:")
    print("  python run_prover.py lookup")
    print("  python run_prover.py gl_quad")

def update_main_nr(experiment):
    """Update main.nr to uncomment the specified experiment's main function."""
    main_nr_path = Path(__file__).parent.parent / "src" / "main.nr"
    
    if not main_nr_path.exists():
        print(f"Error: {main_nr_path} not found")
        return False
    
    # Read the file
    with open(main_nr_path, 'r') as f:
        lines = f.readlines()
    
    # Find all main function blocks and process them
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a "Main function for" comment (including those starting with // //)
        if "Main function for" in line and line.strip().startswith("//"):
            # Check if this is the experiment we want to enable
            target_exp = experiment.upper().replace("_", "")
            should_enable = (target_exp in line.upper()) or (experiment == "gl_quad" and "GL_QUAD" in line.upper())
            
            # Process the next 3 lines (fn main, body, closing brace)
            if i + 1 < len(lines):
                i += 1
                
                if should_enable:
                    # Uncomment this block - remove all leading // and spaces
                    cleaned_line = lines[i].lstrip('/ ').lstrip()
                    lines[i] = cleaned_line
                    
                    if i + 1 < len(lines):
                        i += 1
                        # Remove comment markers and fix indentation
                        cleaned = lines[i].lstrip('/ ').lstrip()
                        lines[i] = '    ' + cleaned if not cleaned.startswith('}') else cleaned
                        
                        if i + 1 < len(lines):
                            i += 1
                            lines[i] = lines[i].lstrip('/ ').lstrip()
                else:
                    # Comment out this block - ensure it has // prefix
                    lines[i] = '// ' + lines[i].lstrip('/ ').lstrip()
                    
                    if i + 1 < len(lines):
                        i += 1
                        cleaned = lines[i].lstrip('/ ').lstrip()
                        lines[i] = '//     ' + cleaned if not cleaned.startswith('}') else '// ' + cleaned
                        
                        if i + 1 < len(lines):
                            i += 1
                            lines[i] = '// ' + lines[i].lstrip('/ ').lstrip()
        i += 1
    
    # Write back
    with open(main_nr_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Updated main.nr to use {experiment.upper()} experiment")
    return True

def run_nargo_execute():
    """Run nargo execute and check if it succeeded."""
    print("\nRunning nargo execute...")
    try:
        result = subprocess.run(
            ["nargo", "execute"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Combine stdout and stderr
        output = result.stdout + result.stderr
        
        # Check for success phrase
        if "Circuit witness successfully solved" in output:
            return True
        else:
            print("nargo execute failed.")
            print("\nOutput:")
            print(output)
            return False
            
    except subprocess.TimeoutExpired:
        print("Failed: nargo execute timed out after 60 seconds")
        return False
    except FileNotFoundError:
        print("Failed: 'nargo' command not found")
        return False
    except Exception as e:
        print(f"Failed: Error running nargo execute: {e}")
        return False

def run_bb_prove():
    """Run bb prove and extract proving key computation time and prover time."""
    print("\nRunning bb prove...")
    try:
        result = subprocess.run(
            [
                "bb", "prove",
                "-b", "./target/non_linear_approx.json",
                "-w", "./target/non_linear_approx.gz",
                "--write_vk",
                "-o", "./target",
                "--print_bench"
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Combine stdout and stderr
        output = result.stdout + result.stderr
        
        # Extract proving key computation time
        # Looking for pattern like: "CircuitProve: Proving key computed in 56 ms (mem: 107.05 MiB)"
        pk_match = re.search(r'Proving key computed in ([\d.]+)\s*ms', output)
        proving_key_time = None
        if pk_match:
            proving_key_time = float(pk_match.group(1))
        
        # Extract prover time
        # Looking for pattern like: "Total: 46 functions (6 shared), 434 measurements, 470.46 ms"
        prover_match = re.search(r'Total:.*?([\d.]+)\s*ms', output)
        prover_time = None
        if prover_match:
            prover_time = float(prover_match.group(1))
        
        # Check if command succeeded
        if result.returncode != 0:
            print("✗ bb prove failed")
            print("\nOutput:")
            print(output)
            return None, None
        
        # Print results
        print("✓ bb prove completed successfully")
        if proving_key_time is not None:
            print(f"  Proving key computation time: {proving_key_time} ms")
        else:
            print("  Warning: Could not extract proving key computation time")
        
        if prover_time is not None:
            print(f"  Prover time: {prover_time} ms")
        else:
            print("  Warning: Could not extract prover time")
        
        return proving_key_time, prover_time
            
    except subprocess.TimeoutExpired:
        print("✗ Failed: bb prove timed out after 5 minutes")
        return None, None
    except FileNotFoundError:
        print("✗ Failed: 'bb' command not found")
        return None, None
    except Exception as e:
        print(f"✗ Failed: Error running bb prove: {e}")
        return None, None

def run_bb_verify():
    """Run bb verify and extract verifier time."""
    print("\nRunning bb verify...")
    try:
        # Use time command via shell to get timing information
        # Run with explicit shell invocation to capture time output
        result = subprocess.run(
            "/usr/bin/time -p bb verify -v -p ./target/proof -k ./target/vk 2>&1",
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            shell=True,
            timeout=60
        )
        
        # Output contains both bb verify output and time stats
        output = result.stdout + result.stderr
        
        # Extract real time from POSIX time format: "real 0.03"
        time_match = re.search(r'real\s+([\d.]+)', output)
        print(time_match)
        verifier_time = None
        if time_match:
            # Convert seconds to milliseconds
            verifier_time = float(time_match.group(1)) * 1000
        else:
            # Try zsh time format: "0.03s user 0.00s system 96% cpu 0.033 total"
            time_match = re.search(r'([\d.]+)\s+total', output)
            if time_match:
                verifier_time = float(time_match.group(1)) * 1000
        
        # Check if command succeeded (look for success message)
        if "Proof verified successfully" not in output and "verified: 1" not in output:
            print("✗ bb verify failed")
            print("\nOutput:")
            print(output)
            return None
        
        # Print result
        print("✓ bb verify completed successfully")
        if verifier_time is not None:
            print(f"  Verifier time: {verifier_time:.2f} ms")
        else:
            print("  Warning: Could not extract verifier time")
        
        return verifier_time
            
    except subprocess.TimeoutExpired:
        print("✗ Failed: bb verify timed out after 60 seconds")
        return None
    except FileNotFoundError:
        print("✗ Failed: 'bb' or 'time' command not found")
        return None
    except Exception as e:
        print(f"✗ Failed: Error running bb verify: {e}")
        return None

def get_proof_size():
    """Get the proof size using du command."""
    print("\nGetting proof size...")
    try:
        result = subprocess.run(
            ["du", "-sh", "./target/proof"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Extract size from output like "16K    ./target/proof"
        output = result.stdout.strip()
        size_match = re.search(r'^([\d.]+[KMG]?)\s+', output)
        proof_size = None
        if size_match:
            proof_size = size_match.group(1)
        
        # Check if command succeeded
        if result.returncode != 0:
            print("✗ Failed to get proof size")
            return None
        
        # Print result
        print("✓ Proof size retrieved")
        if proof_size is not None:
            print(f"  Proof size: {proof_size}")
        else:
            print("  Warning: Could not extract proof size")
        
        return proof_size
            
    except subprocess.TimeoutExpired:
        print("✗ Failed: du command timed out")
        return None
    except FileNotFoundError:
        print("✗ Failed: 'du' command not found")
        return None
    except Exception as e:
        print(f"✗ Failed: Error getting proof size: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    experiment = sys.argv[1].lower()
    
    if experiment in ["-h", "--help", "help"]:
        print_usage()
        sys.exit(0)
    
    if experiment not in EXPERIMENT_MAP:
        print(f"Error: Unknown experiment '{experiment}'")
        print()
        print_usage()
        sys.exit(1)
    
    # Update main.nr
    if not update_main_nr(experiment):
        sys.exit(1)
    
    # Run nargo execute
    if not run_nargo_execute():
        sys.exit(1)
    
    # Run bb prove
    pk_time, prover_time = run_bb_prove()
    if pk_time is None or prover_time is None:
        print("\n⚠ Warning: bb prove completed but could not extract all metrics")
        # Don't exit with error, as the command may have succeeded
    
    # Run bb verify
    verifier_time = run_bb_verify()
    
    # Get proof size
    proof_size = get_proof_size()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment.upper()}")
    print(f"{'='*60}")
    if pk_time is not None:
        print(f"Proving Key Time: {pk_time} ms")
    if prover_time is not None:
        print(f"Prover Time:      {prover_time} ms")
    if verifier_time is not None:
        print(f"Verifier Time:    {verifier_time:.2f} ms")
    if proof_size is not None:
        print(f"Proof Size:       {proof_size}")
    print(f"{'='*60}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
