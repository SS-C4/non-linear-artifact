"""
Generate witness data for fixed-point multiplication verification.
"""
import sys
import toml
import mpmath

mpmath.mp.dps = 100  # set decimal places for mpmath
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def quantize(value, scale):
    """Quantize a value to fixed-point representation."""
    return int(mpmath.nint(mpmath.mpf(value) * scale))

def generate_fp_mul_witness(num_reps, log_scale):
    """Generate witness data for fixed-point multiplication checks."""
    scale = mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale))
    
    witnesses = []
    for _ in range(num_reps):
        # Generate random fixed-point values in [0, 1) with arbitrary precision
        a_real = mpmath.mpf(mpmath.rand())  # Random value in [0, 1)
        b_real = mpmath.mpf(mpmath.rand())  # Random value in [0, 1)
        
        # Compute the product
        c_real = a_real * b_real
        
        # Quantize to fixed-point
        # a represents a_real * S
        # b represents b_real * S
        # c represents c_real * S = (a_real * b_real) * S
        a = quantize(a_real, scale)
        b = quantize(b_real, scale)
        c = quantize(c_real, scale)
        
        # Map to field if negative (shouldn't happen here, but for consistency)
        witnesses.append({
            'a': str(int(a) % field_order),
            'b': str(int(b) % field_order),
            'c': str(int(c) % field_order),
        })
    
    return witnesses

def write_constants(num_reps, log_scale):
    """Write constants.nr file."""
    scale = mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale))
    
    constants = f"""// Number of multiplication checks to perform
pub global NUM_REPS: u32 = {num_reps};

// Scaling factor (2^LOG_SCALE)
pub global LOG_SCALE: u32 = {log_scale};
pub global SCALE: Field = {int(scale)};
"""
    
    with open('src/fp_mul/constants.nr', 'w') as f:
        f.write(constants)

def write_prover_toml(witnesses):
    """Write Prover.toml file."""
    prover_data = {'fp_mul_wits': witnesses}
    
    with open('Prover.toml', 'w') as f:
        toml.dump(prover_data, f)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fp_mul_gen.py <num_reps> <log_scale>")
        print("Example: python fp_mul_gen.py 100 64")
        sys.exit(1)
    
    num_reps = int(sys.argv[1])
    log_scale = int(sys.argv[2])
    
    print(f"Generating {num_reps} fixed-point multiplication witnesses with log_scale={log_scale}")
    
    # Write constants
    write_constants(num_reps, log_scale)
    print("Written constants.nr")
    
    # Generate witnesses
    witnesses = generate_fp_mul_witness(num_reps, log_scale)
    
    # Write Prover.toml
    write_prover_toml(witnesses)
    print("Written Prover.toml")
    
    print("Done!")
