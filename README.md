# Non-linear Approximations
blah blah

## WIPs
- [ ] Numerical integration
- [ ] Padé approximant
- [ ] Lookup table 

## Running the code
- `noirup --version nightly` to get the nightly build. The code is tested with verion 1.0.0.beta.7
- `nargo check (--overwrite)` to (re)generate `Prover.toml`
- Enter the witness element in `Prover.toml`
- `nargo execute` for witness generation
- `bb prove -b ./target/non_linear_approx.json -w ./target/non_linear_approx.gz -o ./target -d` to run the backend and get the circuit size and proving time.