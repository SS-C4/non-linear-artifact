import sys
import toml
import mpmath

mpmath.mp.dps = 100
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

function_map = {
    "exp": "ExponentialWitness",
    "inv_exp": "InverseExponentialWitness",
    "sigmoid": "SigmoidWitness",
    "tanh": "TanhWitness",
    "tan": "TangentWitness",
    "pow": "PowerWitness"
}

def quantize(value, scale):
    return int(mpmath.nint(mpmath.mpf(value) * scale))

def gauss_legendre(n):
    # Gauss-Legendre roots and weights calculation
    roots, weights = mpmath.gauss_quadrature(n)
    
    # Convert roots and weights to strings with precision of 100 decimal places
    roots = [mpmath.nstr(root, n=100) for root in roots]
    weights = [mpmath.nstr(weight, n=100) for weight in weights]
    
    return roots, weights

if __name__ == "__main__":
    if len(sys.argv) >= 6:
        # Parse the arguments when there are 6 or more arguments
        function = sys.argv[1]
        num_inputs = sys.argv[2]
        n_points = int(sys.argv[3])
        log_scale = int(sys.argv[4])
        k = int(sys.argv[5])

        if function != "power":
            k = None

    elif len(sys.argv) == 5:
        # Parse the arguments when there are exactly 5 arguments
        function = sys.argv[1]
        num_inputs = sys.argv[2]
        n_points = int(sys.argv[3])
        log_scale = int(sys.argv[4])
        k = None

        if function == "power":
            print("Error: The 'power' function requires the additional parameter k.")
            sys.exit(1)

    else:
        print("Usage: python3 gl_gen.py <function> <num_inputs> <n_points> <log_scale> [<k> if function is 'power']")
        print("\nArguments:")
        print("  <function>        : Available options are 'exp', 'inv_exp', 'sigmoid', 'tanh', 'tan', and 'power'.")
        print("  <num_inputs>      : The number of inputs required for the function.")
        print("  <n_points>        : The number of points for GL quadrature (integer).")
        print("  <log_scale>       : Log scale for quantization (integer).")
        print("  <k>               : If function is 'power', provide an additional parameter k.")
        sys.exit(1)

    roots, weights = gauss_legendre(n_points)
    roots_tan = [(mpmath.mpf(roots[i]) + 1)**2 / 2 for i in range(n_points)]
    scale = 2 ** log_scale

    # Quantize roots and weights
    quantized_weights = [quantize(w, scale) for w in weights]
    quantized_roots = [quantize(r, scale) for r in roots]
    quantized_roots_tan = [quantize(r, scale) for r in roots_tan]

    # If negative, map to field
    quantized_weights = [str(int(mpmath.mpf(w)) % field_order) for w in quantized_weights]
    quantized_roots = [str(int(mpmath.mpf(r)) % field_order) for r in quantized_roots]
    quantized_roots_tan = [str(int(mpmath.mpf(r)) % field_order) for r in quantized_roots_tan]

    with open("src/gl_quad/constants.nr", "w") as f:
        f.write(f"pub type FUNC_TYPE = super::structs::{function_map[function]};\n")
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global LOG_S: u32 = {log_scale};\n")
        f.write(f"pub global S: Field = {scale};\n")
        f.write(f"pub global S_sq: Field = {scale * scale};\n")
        f.write(f"pub global N_POINTS: u32 = {n_points};\n")
        f.write("pub global GL_ROOTS: [Field; N_POINTS] = [\n")
        for r in quantized_roots:
            f.write(f"    {int(mpmath.mpf(r))},\n")
        f.write("];\n")
        f.write("pub global GL_ROOTS_tan: [Field; N_POINTS] = [\n")
        for r in quantized_roots_tan:
            f.write(f"    {int(mpmath.mpf(r))},\n")
        f.write("];\n")
        f.write("pub global GL_WEIGHTS: [Field; N_POINTS] = [\n")
        for w in quantized_weights:
            f.write(f"    {int(mpmath.mpf(w))},\n")
        f.write("];\n")

    # Witness generation
    # y = f(x)

    gl_wits = []
    for i in range(int(num_inputs)):
        x_input = mpmath.mpf(mpmath.rand())
        
        if function == "exp":
            y_input = mpmath.exp(x_input)
        elif function == "inv_exp":
            y_input = 1 / mpmath.exp(x_input)
        elif function == "sigmoid":
            y_input = 1 / (1 + mpmath.exp(-x_input))
        elif function == "tanh":
            y_input = mpmath.tanh(x_input)
        elif function == "tan":
            y_input = mpmath.tan(x_input)
        elif function == "pow":
            if k is None:
                print("Error: The 'power' function requires the additional parameter k.")
                sys.exit(1)
            y_input = mpmath.power(x_input, k)
        else:
            print(f"Error: Unknown function '{function}'.")
            sys.exit(1)
        
        x_quantized = quantize(x_input, scale)
        y_quantized = quantize(y_input, scale)

        k_input = mpmath.mpf(k) if k is not None else None
        k_quantized = quantize(k_input, scale) if k_input is not None else None

        if function == "exp":
            ratio = (y_input + 1) / (y_input - 1)
            
            gl_inverses = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = 1 / (r + ratio)
                gl_inverses.append(quantize(gl_inverse, scale))
                
            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "ratio_term": str(quantize(ratio, scale)),
                    "denom_inverses": [str(v) for v in gl_inverses]
                }
            })
            
        elif function == "inv_exp":
            ratio = (1 + y_input) / (1 - y_input)

            gl_inverses = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = 1 / (r + ratio)
                gl_inverses.append(quantize(gl_inverse, scale))

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "ratio_term": str(quantize(ratio, scale)),
                    "denom_inverses": [str(v) for v in gl_inverses]
                }
            })

        elif function == "sigmoid":
            ratio = 1 / (2 * y_input - 1)

            gl_inverses = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = 1 / (r + ratio)
                gl_inverses.append(quantize(gl_inverse, scale))

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "ratio_term": str(quantize(ratio, scale)),
                    "denom_inverses": [str(v) for v in gl_inverses]
                }
            })

        elif function == "tanh":
            ratio = 2 / y_input

            gl_inverses = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = 1 / (2 * r + ratio)
                gl_inverses.append(quantize(gl_inverse, scale))

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "ratio_term": str(quantize(ratio, scale)),
                    "denom_inverses": [str(v) for v in gl_inverses]
                }
            })

        elif function == "tan":
            ratio = 2 / y_input

            gl_inverses = []
            mult_terms = []
            for r in roots_tan:
                r = mpmath.mpf(r)
                mult_term = y_input / 2 * (r + 1)**2
                mult_terms.append(quantize(mult_term, scale))
                gl_inverse = 1 / (y_input / 2 * (r + 1)**2 + ratio)
                gl_inverses.append(quantize(gl_inverse, scale))

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "ratio_term": str(quantize(ratio, scale)),
                    "denom_inverses": [str(v) for v in gl_inverses],
                    "mult_terms": [str(v) for v in mult_terms]
                }
            })
        
        elif function == "pow":
            ratio_1 = (1 + y_input) / (1 - y_input)
            ratio_2 = (1 + x_input) / (1 - x_input)

            gl_inverses_1 = []
            gl_inverses_2 = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse_1 = 1 / (r + ratio_1)
                gl_inverse_2 = 1 / (r + ratio_2)
                gl_inverses_1.append(quantize(gl_inverse_1, scale))
                gl_inverses_2.append(quantize(gl_inverse_2, scale))

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "ratio_term_1": str(quantize(ratio_1, scale)),
                    "ratio_term_2": str(quantize(ratio_2, scale)),
                    "denom_inverses_1": [str(v) for v in gl_inverses_1],
                    "denom_inverses_2": [str(v) for v in gl_inverses_2]
                }
            })

    with open("Prover.toml", "r+") as f:
        toml_data = toml.load(f)
        toml_data["_gl_wits"] = gl_wits
        f.seek(0)
        toml.dump(toml_data, f)
        f.truncate()
