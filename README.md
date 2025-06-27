# Non-linear Approximations
blah blah

## WIPs
- [ ] Numerical integration
- [ ] Padé approximant
- [ ] Lookup table 

## Running the code
- `noirup --version 1.0.0-beta.0`
- `nargo check (--overwrite)` to (re)generate `Prover.toml`
- Enter the witness elements in `Prover.toml`
- `nargo execute` for witness generation
- `bb prove -b ./target/simpson.json -w ./target/simpson.gz -o ./target -d` to run the backend and get the circuit size / proving time etc.