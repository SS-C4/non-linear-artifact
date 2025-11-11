import sys
import toml
import mpmath

mpmath.mp.dps = 100
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

function_map = {
    "exp": mpmath.exp,
    "cos": mpmath.cos,
}

def quantize(value, scale):
    return int(mpmath.nint(mpmath.mpf(value) * scale))

def taylor_coeffs(func, degree, log_scale):
    # Centered at 0
    scale = 2 ** log_scale
    coeffs = []
    for n in range(degree + 1):
        # Get the nth derivative at x=0
        coeff = mpmath.diff(func, 0, n) / mpmath.factorial(n)
        coeffs.append(quantize(coeff, scale) % field_order)
    return coeffs

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python poly_gen.py <function> <num_inputs> <degree> <log_scale>")
        sys.exit(1)

    func_name = sys.argv[1]
    num_inputs = int(sys.argv[2])
    degree = int(sys.argv[3])
    log_scale = int(sys.argv[4])

    if func_name not in function_map:
        print(f"Unknown function: {func_name}")
        sys.exit(1)

    func = function_map[func_name]
    coeffs = taylor_coeffs(func, degree, log_scale)

    with open("src/poly_approx/constants.nr", "w") as f:
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global LOG_SCALE: u32 = {log_scale};\n")
        f.write(f"pub global SCALE: Field = {2 ** log_scale};\n")
        f.write(f"pub global DEGREE: u32 = {degree};\n")
        f.write(f"pub global COEFFS: [Field; {degree + 1}] = [\n")
        for coeff in coeffs:
            f.write(f"    {coeff},\n")
        f.write("];\n")

    poly_wits = []
    for i in range(num_inputs):
        x_input = mpmath.mpf(mpmath.rand())
        x_quantized = quantize(x_input, 2 ** log_scale)

        y = func(x_input)
        y_quantized = quantize(y, 2 ** log_scale)

        intermediates = []
        for j in range(degree + 1):
            x_power = x_input ** j
            x_power_quantized = quantize(x_power, 2 ** log_scale)
            intermediates.append(x_power_quantized % field_order)

        poly_wits.append({
            "x": str(x_quantized % field_order),
            "y": str(y_quantized % field_order),
            "intermediates": [str(val) for val in intermediates]
        })

    with open("Prover.toml", "r+") as f:
        toml_data = toml.load(f)
        toml_data["poly_wits"] = poly_wits
        f.seek(0)
        toml.dump(toml_data, f)
        f.truncate()
