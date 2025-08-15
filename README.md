# Non-linear Approximations
blah blah

## Running the code
- `pip3 install -r requirements.txt` to install necessary packages
- `noirup` to get the latest version of Noir. The code is tested with version 1.0.0.beta.6

### 1. Newton-Cotes Based Approximation

`
python3 scripts/nc_gen.py <input> <order> <log_scale>
`
**Arguments:**

- `<input>`: Name of the function (e.g., `exp`, `sigmoid`)
- `<order>`: Order of Newton-Cotes integration (e.g., `2`, `4`, `6`)
- `<log_scale>`: Log scale for quantization

Populates `Prover.toml` and `src/nc_int/constants.nr` with the appropriate values.

---

### 2. Lookup-Based Protocol

`
python3 scripts/exp_gen.py <func> <num_inputs> <log_base> <log_scale> [<softmax_size> if func is 'softmax']
`

**Arguments:**

- `<func>`: Function type, either `inv_exp` or `softmax`
- `<num_inputs>`: Number of inputs to the function
- `<log_base>`: Log base for the size of each lookup table
- `<log_scale>`: Log scale for quantization
- `<softmax_size>` *(only if `func` is `softmax`)*: Dimension of the softmax input vector

Populates `Prover.toml` and `src/exp_int/constants.nr`.

---

### 3. Gauss-Legendre Based Approximation

`
python3 scripts/gl_gen.py <function> <num_inputs> <n_points> <log_scale> [<k> if function is 'power'] [<dim_softmax> if function is 'softmax']
`

**Arguments:**

- `<function>`: One of `'exp'`, `'inv_exp'`, `'sigmoid'`, `'tanh'`, `'tan'`, `'power'`, `'softmax'`
- `<num_inputs>`: Number of function inputs
- `<n_points>`: Number of quadrature points
- `<log_scale>`: Log scale for quantization
- `<k>` *(required if function is `'power'`)*: Exponent value for `x^k`
- `<dim_softmax>` *(required if function is `'softmax'`)*: Size of the softmax input vector

Populates constants for Gauss-Legendre based approximations.

---

## Common Instructions

- **Generate witness**:
  `
  nargo execute
  `

- **Run the backend to get circuit size and proving time**:
  `
  bb prove -b ./target/non_linear_approx.json -w ./target/non_linear_approx.gz -o ./target -d
  `
