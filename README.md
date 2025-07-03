# Non-linear Approximations
blah blah

## WIPs
- [ ] Numerical integration
- [ ] Padé approximant
- [ ] Lookup table 

## Running the code
- `noirup --version nightly` to get the nightly build. The code is tested with verion 1.0.0.beta.7
- Set the Newton-Cotes integration order in `main.nr`
- `python3 wit-gen.py <input> <order[must be odd]> <scale>` to (re)generate `Prover.toml`
- `nargo execute` for witness generation
- `bb prove -b ./target/non_linear_approx.json -w ./target/non_linear_approx.gz -o ./target -d` to run the backend and get the circuit size and proving time