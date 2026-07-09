# Non-linear Approximations

This project implements and benchmarks various approximation methods for non-linear functions in zero-knowledge circuits using Noir. This repository is the reference implementation for the paper https://eprint.iacr.org/2025/2326.

## Setup

Requires Python 3.13+.

```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Install Noir and Barretenberg (tested with latest stable (1.0.0-beta.21))
noirup
bbup
```

## Quick Start

### Automated Benchmarking (Recommended)

Run comprehensive experiments across all functions and approximation methods:

```bash
# Run all experiments
python3 scripts/run_all.py

# Run experiments for a specific function
python3 scripts/run_all.py --function sigmoid

# Run a specific function with a specific approximation method
python3 scripts/run_all.py --function sigmoid --experiment poly

# List available options
python3 scripts/run_all.py --list
```

Results are saved to `experiments/results_*.csv` with timing data and proof sizes.

### Manual Workflow

For individual experiments:

1. **Generate witness** using one of the generator scripts (see below)
2. **Run the prover**: `python3 scripts/run_prover.py <experiment_type>`

Example:
```bash
python3 scripts/poly_gen.py sigmoid 100 6 32
python3 scripts/run_prover.py poly
```

---

## Available Functions

- `inv_exp` - Inverse exponential (1/e^x)
- `sigmoid` - Sigmoid activation function
- `gelu` - GELU activation function
- `erf` - Error function
- `tanh` - Hyperbolic tangent
- `tan` - Tangent
- `cos` - Cosine
- `power` - Power function (x^k)

---

## Approximation Methods

### 1. Polynomial Approximation (Taylor Series)

```bash
python3 scripts/poly_gen.py <function> <num_inputs> <degree> <log_scale>
python3 scripts/poly_gen.py power <num_inputs> <degree> <log_scale> <k>
```

**Arguments:**
- `<function>`: Function name (inv_exp, cos, tan, sigmoid, tanh, gelu, erf, power)
- `<num_inputs>`: Number of test inputs
- `<degree>`: Degree of the Taylor polynomial
- `<log_scale>`: Log₂ of the fixed-point scale
- `<k>`: Exponent for power function only

**Example:**
```bash
python3 scripts/poly_gen.py sigmoid 100 6 32
python3 scripts/poly_gen.py power 100 6 32 0.876
```

---

### 2. Padé Approximation (Rational Function)

```bash
python3 scripts/pade_gen.py <function> <num_inputs> <degree> <log_scale>
python3 scripts/pade_gen.py power <num_inputs> <degree> <log_scale> <k>
```

**Arguments:**
- Same as polynomial approximation
- Produces rational function approximation (numerator/denominator)

**Example:**
```bash
python3 scripts/pade_gen.py sigmoid 100 6 32
```

---

### 3. Piecewise Linear Approximation (PWL)

```bash
python3 scripts/pwl_gen.py <function> <num_inputs> <x_start> <x_end> <log_num_pieces> <log_scale>
python3 scripts/pwl_gen.py power <num_inputs> <x_start> <x_end> <log_num_pieces> <log_scale> <k>
```

**Arguments:**
- `<x_start>`, `<x_end>`: Domain boundaries
- `<log_num_pieces>`: Log₂ of the number of linear pieces
- Other arguments same as above

**Example:**
```bash
python3 scripts/pwl_gen.py sigmoid 100 0.0 1.0 4 32
```

---

### 4. Gauss-Legendre Quadrature

```bash
python3 scripts/gl_gen.py <function> <num_inputs> <n_points> <log_scale> [<k>] [<dim_softmax>]
```

**Arguments:**
- `<function>`: exp, inv_exp, sigmoid, tanh, tan, cos, power, softmax, gelu, erf
- `<n_points>`: Number of quadrature points
- `<k>`: Required for power function
- `<dim_softmax>`: Required for softmax function

**Example:**
```bash
python3 scripts/gl_gen.py sigmoid 100 8 32
python3 scripts/gl_gen.py power 100 8 32 0.876
```

---

### 5. Lookup Table-Based Approximation

```bash
python3 scripts/lookup_gen.py <function> <num_inputs> <log_base> <log_scale> [<k>] [<dim_softmax>]
```

**Arguments:**
- `<log_base>`: Log₂ of the lookup table base size
- Other arguments similar to Gauss-Legendre
- For cos/tan: log_base must equal log_scale

**Example:**
```bash
python3 scripts/lookup_gen.py sigmoid 100 4 32
python3 scripts/lookup_gen.py power 100 4 32 0.876
```

---

## Running the Prover

After generating a witness with any of the above scripts:

```bash
python3 scripts/run_prover.py <experiment_type>
```

Where `<experiment_type>` is one of:
- `poly` - Polynomial approximation
- `pade` - Padé approximation
- `pwl` - Piecewise linear
- `gl_quad` - Gauss-Legendre quadrature
- `lookup` - Lookup tables

**Example:**
```bash
python3 scripts/run_prover.py poly
```

This will:
1. Update `src/main.nr` to enable the correct experiment
2. Run `nargo execute` to generate the circuit witness
3. Run `bb prove` to generate the proof
4. Run `bb verify` to verify the proof
5. Report proving key time, prover time, verifier time, and proof size

---

## Experiment Results

Results from `run_all.py` are stored in the `experiments/` directory as CSV files.

### File Naming Convention

- `results_all_YYYYMMDD_HHMMSS.csv` - All experiments
- `results_<function>_YYYYMMDD_HHMMSS.csv` - Single function, all methods
- `results_<function>_<experiment>_YYYYMMDD_HHMMSS.csv` - Single function and method

### CSV Columns

- **function**: Mathematical function (sigmoid, gelu, etc.)
- **experiment**: Approximation method (poly, pade, pwl, gl_quad, lookup)
- **log_scale**: Log₂ of the fixed-point scale
- **num_inputs**: Number of test inputs
- **params**: Full parameter list
- **pk_time_ms**: Proving key computation time (ms)
- **prover_time_ms**: Prover execution time (ms)
- **verifier_time_ms**: Verifier execution time (ms)
- **proof_size**: Proof size in bytes

---

## Configuration

The automated script `run_all.py` uses the following defaults:

- **NUM_INPUTS**: 100
- **LOG_SCALES**: [16, 32, 64, 120]

### Parameter Scaling

**Polynomial & Padé:**
- Degree doubles with each log_scale increase: 6 → 12 → 24 → 48

**Gauss-Legendre:**
- Points increase linearly: 8 → 12 → 16 → 20

**Lookup Tables:**
- Log_base fixed at 8 for decomposable tables
- Log_base equals log_scale for cos/tan (full precision tables)

Edit `scripts/run_all.py` to customize parameters for each function/method/scale combination.

---

## Examples

### Compare All Approximation Methods for Sigmoid

```bash
python3 scripts/run_all.py --function sigmoid
```

### Benchmark Polynomial Approximations Across Functions

```bash
python3 scripts/run_all.py --function inv_exp --experiment poly
python3 scripts/run_all.py --function sigmoid --experiment poly
python3 scripts/run_all.py --function gelu --experiment poly
```

### Run a Single Custom Experiment

```bash
# Generate witness
python3 scripts/poly_gen.py tanh 50 8 64

# Run prover
python3 scripts/run_prover.py poly
```

---

## Notes

- Results are saved incrementally (safe to interrupt with Ctrl+C)
- Failed experiments are marked as "FAILED" in CSV output
- Each automated run creates a new timestamped CSV file
- Use `--list` flag to see all available functions and experiment types
