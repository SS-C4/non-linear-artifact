import sys
import toml
import mpmath

mpmath.mp.dps = 100
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

function_map = {
    "exp": mpmath.exp,
    "cos": mpmath.cos,
    "tan": mpmath.tan,
    "sigmoid": lambda x: 1 / (1 + mpmath.exp(-x)),
    "tanh": mpmath.tanh,
}

def quantize(value, scale):
    return int(mpmath.nint(mpmath.mpf(value) * scale))

def pade_coeffs(func, degree, log_scale):
    # Centered at 0
    scale = 2 ** log_scale
    
    taylor_coeffs = []
    for n in range(2 * degree + 1):  # Compute enough coefficients for the Pade approximant
        coeff = mpmath.diff(func, 0, n) / mpmath.factorial(n)
        taylor_coeffs.append(coeff)

    p, q = mpmath.pade(taylor_coeffs, degree, degree)

    num = []
    denom = []
    for i in range(len(p)):
        num.append(quantize(p[i], scale) % field_order)
    for i in range(len(q)):
        denom.append(quantize(q[i], scale) % field_order)

    return num, denom, q


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python pade_gen.py <function> <num_inputs> <degree> <log_scale>")
        sys.exit(1)

    func_name = sys.argv[1]
    num_inputs = int(sys.argv[2])
    degree = int(sys.argv[3])
    log_scale = int(sys.argv[4])

    if func_name not in function_map:
        print(f"Unknown function: {func_name}")
        sys.exit(1)

    func = function_map[func_name]
    num, denom, q = pade_coeffs(func, degree, log_scale)

    with open("src/pade_approx/constants.nr", "w") as f:
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global LOG_SCALE: u32 = {log_scale};\n")
        f.write(f"pub global SCALE: Field = {2 ** log_scale};\n")
        f.write(f"pub global DEGREE: u32 = {degree};\n")
        f.write(f"pub global NUM_COEFFS: [Field; {degree + 1}] = [\n")
        for coeff in num:
            f.write(f"    {coeff},\n")
        f.write("];\n")
        f.write(f"pub global DENOM_COEFFS: [Field; {degree + 1}] = [\n")
        for coeff in denom:
            f.write(f"    {coeff},\n")
        f.write("];\n")

    pade_wits = []
    for i in range(num_inputs):
        x_input = mpmath.mpf(mpmath.rand())
        x_quantized = quantize(x_input, 2 ** log_scale)

        y = func(x_input)
        y_quantized = quantize(y, 2 ** log_scale)

        intermediates = []
        denom_value = mpmath.mpf(0)
        for j in range(degree + 1):
            x_power = x_input ** j
            denom_value += q[j] * x_power

            x_power_quantized = quantize(x_power, 2 ** log_scale)
            intermediates.append(x_power_quantized % field_order)

        pade_wits.append({
            "x": str(x_quantized % field_order),
            "y": str(y_quantized % field_order),
            "intermediates": [str(val) for val in intermediates],
            "denom_rescaled": str(quantize(denom_value, 2 ** log_scale) % field_order)
        })

    with open("Prover.toml", "r+") as f:
        toml_data = toml.load(f)
        toml_data["pade_wits"] = pade_wits
        f.seek(0)
        toml.dump(toml_data, f)
        f.truncate()