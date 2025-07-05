# Non-linear Approximations
blah blah

## WIPs
- [x] Numerical integration
- [ ] Padé approximant
- [ ] Custom lookup table 

## Running the code
- `pip3 install -r requirements.txt` to install necessary packages
- `noirup` to get the latest version of Noir. The code is tested with version 1.0.0.beta.6
  
For Numerical integration
- `python3 scripts/nc_gen.py <input> <order for NC integration approx> <log_scale for quantization>` to (re)generate `Prover.toml` and `src/nc_int/constants.nr` with the appropriate advice and weights respectively.
- `nargo execute` for witness generation
- `bb prove -b ./target/non_linear_approx.json -w ./target/non_linear_approx.gz -o ./target -d` to run the backend and get the circuit size and proving time

For Lookup based protocol
- TODO