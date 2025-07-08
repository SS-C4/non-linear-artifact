# Non-linear Approximations
blah blah

## Running the code
- `pip3 install -r requirements.txt` to install necessary packages
- `noirup` to get the latest version of Noir. The code is tested with version 1.0.0.beta.6
  
For Newton-Cotes based approximation:
- `python3 scripts/nc_gen.py <input> <order for NC integration approx> <log_scale for quantization>` to populate `Prover.toml` and `src/nc_int/constants.nr` with the appropriate values depending on the parameters.

For Lookup-based protocol:
- `python3 scripts/exp_gen.py <input> <log_base for size of each lookup table> <log_scale for quantization>` to populate `Prover.toml` and `src/exp_int/constants.nr` with the appropriate values depending on the parameters.

Common instructions:
- `nargo execute` for witness generation
- `bb prove -b ./target/non_linear_approx.json -w ./target/non_linear_approx.gz -o ./target -d` to run the backend and get the circuit size and proving time (note that this contains both of the above protocols)