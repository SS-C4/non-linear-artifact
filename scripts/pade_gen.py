import sys
import toml
import mpmath

mpmath.mp.dps = 100
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

function_map = {
    "inv_exp": lambda x: mpmath.mpf(1) / mpmath.exp(x),
    "cos": mpmath.cos,
    "tan": mpmath.tan,
    "sigmoid": lambda x: mpmath.mpf(1) / (mpmath.mpf(1) + mpmath.exp(-x)),
    "tanh": mpmath.tanh,
    "gelu": lambda x: mpmath.mpf('0.5') * x * (mpmath.mpf(1) + mpmath.tanh(mpmath.sqrt(mpmath.mpf(2) / mpmath.pi) * (x + mpmath.mpf('0.044715') * mpmath.power(x, mpmath.mpf(3))))),
    "erf": mpmath.erf,
    "power": mpmath.exp,
    "kepler": mpmath.cos,
}

def quantize(value, scale):
    return int(mpmath.nint(mpmath.mpf(value) * scale))

def pade_coeffs(func, degree, log_scale):
    # Centered at 0
    scale = mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale))
    
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
    if len(sys.argv) < 5:
        print("Usage: python pade_gen.py <function> <num_inputs> <degree> <log_scale>")
        print("       python pade_gen.py power <num_inputs> <degree> <log_scale> <k>")
        print("\nAvailable functions: inv_exp, cos, tan, sigmoid, tanh, gelu, erf, power")
        print("\nExample:")
        print("  python pade_gen.py sigmoid 100 6 32")
        print("  python pade_gen.py power 100 6 32 0.876")
        sys.exit(1)

    func_name = sys.argv[1]
    num_inputs = int(sys.argv[2])
    degree = int(sys.argv[3])
    log_scale = int(sys.argv[4])
    k = None

    if func_name == "power":
        if len(sys.argv) != 6:
            print("Error: power function requires 5 arguments")
            print("Usage: python pade_gen.py power <num_inputs> <degree> <log_scale> <k>")
            print("Example: python pade_gen.py power 100 6 32 0.876")
            sys.exit(1)
        k = mpmath.mpf(sys.argv[5])
    elif len(sys.argv) != 5:
        print(f"Error: {func_name} function requires 4 arguments")
        print("Usage: python pade_gen.py <function> <num_inputs> <degree> <log_scale>")
        sys.exit(1)

    if func_name not in function_map and func_name != "power" and func_name != "kepler":
        print(f"Unknown function: {func_name}")
        print("Available functions: inv_exp, cos, tan, sigmoid, tanh, gelu, erf, power, kepler")
        sys.exit(1)

    func = function_map[func_name]
    num, denom, q = pade_coeffs(func, degree, log_scale)

    with open("src/pade_approx/constants.nr", "w") as f:
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global LOG_SCALE: u32 = {log_scale};\n")
        f.write(f"pub global SCALE: Field = {int(mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale)))};\n")
        f.write(f"pub global DEGREE: u32 = {degree};\n")
        f.write(f"pub global NUM_COEFFS: [Field; {degree + 1}] = [\n")
        for coeff in num:
            f.write(f"    {coeff},\n")
        f.write("];\n")
        f.write(f"pub global DENOM_COEFFS: [Field; {degree + 1}] = [\n")
        for coeff in denom:
            f.write(f"    {coeff},\n")
        f.write("];\n")
        f.write(f"pub global IS_POWER: bool = {str(func_name == "power").lower()};\n")
        f.write(f"pub global IS_KEPLER: bool = {str(func_name == "kepler").lower()};\n")
        f.write(f"pub global TWO_PI: Field = {quantize(mpmath.mpf(2) * mpmath.pi, mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale)))};\n")
        f.write(f"pub global PI: Field = {quantize(mpmath.pi, mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale)))};\n")

    pade_wits = []
    if func_name == "kepler":
        scale = mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale))
        for i in range(num_inputs):
            P = mpmath.mpf(mpmath.rand()) + mpmath.mpf(1)
            e = mpmath.mpf(mpmath.rand()) * mpmath.mpf('0.4') + mpmath.mpf('0.5')
            dt = mpmath.mpf(mpmath.rand()) * P

            a = mpmath.power(P, mpmath.mpf(2) / mpmath.mpf(3))
            a_sq = mpmath.power(a, mpmath.mpf(2))
            e_sq = mpmath.power(e, mpmath.mpf(2))
            sqrt_1_m_e2 = mpmath.sqrt(mpmath.mpf(1) - e_sq)
            b = a * sqrt_1_m_e2
            M = mpmath.mpf(2) * mpmath.pi * dt / P
            E = mpmath.findroot(lambda E: E - e * mpmath.sin(E) - M, M)

            sin_E = mpmath.sin(E)
            cos_E = mpmath.cos(E)
            x = a * (cos_E - e)
            y = b * sin_E

            selector = 0 if E < mpmath.pi / mpmath.mpf(2) else 1 if E < mpmath.mpf(3) * mpmath.pi / mpmath.mpf(2) else 2
            cos_E_shifted = cos_E if selector == 0 else -cos_E if selector == 1 else cos_E
            E_shifted = E - mpmath.mpf(selector) * mpmath.pi
            print(E_shifted)

            intermediates = []
            denom_value = mpmath.mpf(0)
            for j in range(degree + 1):
                E_power = mpmath.power(E_shifted, mpmath.mpf(j))
                denom_value += q[j] * E_power

                E_power_quantized = quantize(E_power, scale)
                intermediates.append(E_power_quantized % field_order)

            pade_wits.append({
                "x": "0",
                "y": "0",
                "intermediates": [str(val) for val in intermediates],
                "denom_rescaled": str(quantize(denom_value, scale) % field_order),
                "k": "0",
                "log_x": "0",
                "k_log_x": "0",
                "intermediates_2": ["0" for _ in range(degree + 1)],
                "denom_rescaled_2": "0",
                "kepler_witness": {
                    "P_orbit": str(quantize(P, scale) % field_order),
                    "dt": str(quantize(dt, scale) % field_order),
                    "a": str(quantize(a, scale) % field_order),
                    "a_sq": str(quantize(a_sq, scale) % field_order),
                    "sqrt_1_m_e2": str(quantize(sqrt_1_m_e2, scale) % field_order),
                    "b": str(quantize(b, scale) % field_order),
                    "E": str(quantize(E, scale) % field_order),
                    "M": str(quantize(M, scale) % field_order),
                    "e": str(quantize(e, scale) % field_order),
                    "sin_E": str(quantize(sin_E, scale) % field_order),
                    "cos_E": str(quantize(cos_E, scale) % field_order),
                    "cos_E_shifted": str(quantize(cos_E_shifted, scale) % field_order),
                    "x": str(quantize(x, scale) % field_order),
                    "y": str(quantize(y, scale) % field_order),
                    "selector": str(selector)
                }
            })

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["pade_wits"] = pade_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func_name != "power":
        scale = mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale))
        for i in range(num_inputs):
            x_input = mpmath.mpf(mpmath.rand())
            x_quantized = quantize(x_input, scale)

            y = func(x_input)
            y_quantized = quantize(y, scale)

            k_quantized = None

            intermediates = []
            denom_value = mpmath.mpf(0)
            for j in range(degree + 1):
                x_power = mpmath.power(x_input, mpmath.mpf(j))
                denom_value += q[j] * x_power

                x_power_quantized = quantize(x_power, scale)
                intermediates.append(x_power_quantized % field_order)

            pade_wits.append({
                "x": str(x_quantized % field_order),
                "y": str(y_quantized % field_order),
                "intermediates": [str(val) for val in intermediates],
                "denom_rescaled": str(quantize(denom_value, scale) % field_order),
                "k": "0",
                "log_x": "0",
                "k_log_x": "0",
                "intermediates_2": ["0" for _ in range(degree + 1)],
                "denom_rescaled_2": "0",
                "kepler_witness": {}
            })

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["pade_wits"] = pade_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    else:
        scale = mpmath.power(mpmath.mpf(2), mpmath.mpf(log_scale))
        for i in range(num_inputs):
            x_input = mpmath.mpf(mpmath.rand()) + mpmath.mpf(1)
            y = mpmath.exp(k * mpmath.log(x_input))
            log_x = mpmath.log(x_input)
            k_log_x = k * log_x

            x_quantized = quantize(x_input, scale)
            y_quantized = quantize(y, scale)
            k_quantized = quantize(k, scale)
            log_x_quantized = quantize(log_x, scale)
            k_log_x_quantized = quantize(k_log_x, scale)

            # For x = e^(log_x)
            intermediates = []
            denom_value = mpmath.mpf(0)
            for j in range(degree + 1):
                x_power = mpmath.power(log_x, mpmath.mpf(j))
                denom_value += q[j] * x_power

                x_power_quantized = quantize(x_power, scale)
                intermediates.append(x_power_quantized % field_order)

            # For y = e^(k * log_x)
            intermediates_2 = []
            denom_value_2 = mpmath.mpf(0)
            for j in range(degree + 1):
                x_power_2 = mpmath.power(k_log_x, mpmath.mpf(j))
                denom_value_2 += q[j] * x_power_2

                x_power_quantized_2 = quantize(x_power_2, scale)
                intermediates_2.append(x_power_quantized_2 % field_order)

            pade_wits.append({
                "x": str(x_quantized % field_order),
                "y": str(y_quantized % field_order),
                "intermediates": [str(val) for val in intermediates],
                "denom_rescaled": str(quantize(denom_value, scale) % field_order),
                "k": str(k_quantized % field_order),
                "log_x": str(log_x_quantized % field_order),
                "k_log_x": str(k_log_x_quantized % field_order),
                "intermediates_2": [str(val) for val in intermediates_2],
                "denom_rescaled_2": str(quantize(denom_value_2, scale) % field_order),
                "kepler_witness": {}
            })

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["pade_wits"] = pade_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()