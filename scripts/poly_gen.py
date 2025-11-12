import sys
import toml
import mpmath

mpmath.mp.dps = 100
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

function_map = {
    "inv_exp": lambda x: 1 / mpmath.exp(x),
    "cos": mpmath.cos,
    "tan": mpmath.tan,
    "sigmoid": lambda x: 1 / (1 + mpmath.exp(-x)),
    "tanh": mpmath.tanh,
    "gelu": lambda x: 0.5 * x * (1 + mpmath.tanh(mpmath.sqrt(2 / mpmath.pi) * (x + 0.044715 * x ** 3))),
    "erf": mpmath.erf,
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
    if len(sys.argv) < 5:
        print("Usage: python poly_gen.py <function> <num_inputs> <degree> <log_scale>")
        print("       python poly_gen.py power <num_inputs> <degree> <log_scale> <k>")
        print("\nAvailable functions: inv_exp, cos, tan, sigmoid, tanh, gelu, erf, power")
        print("\nExample:")
        print("  python poly_gen.py sigmoid 100 6 32")
        print("  python poly_gen.py power 100 6 32 0.876")
        sys.exit(1)

    func_name = sys.argv[1]
    num_inputs = int(sys.argv[2])
    degree = int(sys.argv[3])
    log_scale = int(sys.argv[4])
    k = None

    if func_name == "power":
        if len(sys.argv) != 6:
            print("Error: power function requires 5 arguments")
            print("Usage: python poly_gen.py power <num_inputs> <degree> <log_scale> <k>")
            print("Example: python poly_gen.py power 100 6 32 0.876")
            sys.exit(1)
        k = mpmath.mpf(sys.argv[5])
    elif len(sys.argv) != 5:
        print(f"Error: {func_name} function requires 4 arguments")
        print("Usage: python poly_gen.py <function> <num_inputs> <degree> <log_scale>")
        sys.exit(1)

    if func_name not in function_map and func_name != "power":
        print(f"Unknown function: {func_name}")
        print("Available functions: inv_exp, cos, tan, sigmoid, tanh, gelu, erf, power")
        sys.exit(1)

    func = function_map[func_name] if func_name != "power" else mpmath.exp
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
        f.write(f"pub global IS_POWER: bool = {str(func_name == 'power').lower()};\n")

    poly_wits = []
    if func_name != "power":
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
                "intermediates": [str(val) for val in intermediates],
                "k": "0",
                "log_x": "0",
                "k_log_x": "0",
                "intermediates_2": ["0" for _ in range(degree + 1)]
            })

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["poly_wits"] = poly_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()
    else:
        for i in range(num_inputs):
            x_input = mpmath.mpf(mpmath.rand() + 1)
            x_quantized = quantize(x_input, 2 ** log_scale)

            y = mpmath.exp(k * mpmath.log(x_input))
            y_quantized = quantize(y, 2 ** log_scale)

            log_x = mpmath.log(x_input)
            k_log_x = k * log_x

            log_x_quantized = quantize(log_x, 2 ** log_scale)
            k_log_x_quantized = quantize(k_log_x, 2 ** log_scale)

            intermediates = []
            for j in range(degree + 1):
                x_power = log_x ** j
                x_power_quantized = quantize(x_power, 2 ** log_scale)
                intermediates.append(x_power_quantized % field_order)

            intermediates_2 = []
            for j in range(degree + 1):
                x_power = k_log_x ** j
                x_power_quantized = quantize(x_power, 2 ** log_scale)
                intermediates_2.append(x_power_quantized % field_order)

            poly_wits.append({
                "x": str(x_quantized % field_order),
                "y": str(y_quantized % field_order),
                "intermediates": [str(val) for val in intermediates],
                "k": str(quantize(k, 2 ** log_scale) % field_order),
                "log_x": str(log_x_quantized % field_order),
                "k_log_x": str(k_log_x_quantized % field_order),
                "intermediates_2": [str(val) for val in intermediates_2],
            })

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["poly_wits"] = poly_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()