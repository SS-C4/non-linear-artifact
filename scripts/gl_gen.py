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
    "cos": "CosineWitness",
    "power": "PowerWitness",
    "softmax": "SoftmaxWitness",
    "gelu": "GeluWitness",
    "erf": "ErfWitness"
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
    if len(sys.argv) < 5:
        print("Usage: python3 gl_gen.py <function> <num_inputs> <n_points> <log_scale> [<k> if function is 'power'] [<dim_softmax> if function is 'softmax']")
        print("\nArguments:")
        print("  <function>        : 'exp', 'inv_exp', 'sigmoid', 'tanh', 'tan', 'cos', 'power', 'softmax'.")
        print("  <num_inputs>      : Number of inputs required for the function.")
        print("  <n_points>        : Number of points for GL quadrature (integer).")
        print("  <log_scale>       : Log scale for quantization (integer).")
        print("  <k>               : Required if function is 'power'.")
        print("  <dim_softmax>     : Required if function is 'softmax'.")
        sys.exit(1)

    # Parse required arguments
    function = sys.argv[1]
    num_inputs = sys.argv[2]
    n_points = int(sys.argv[3])
    log_scale = int(sys.argv[4])

    k = None
    dim_softmax = None

    # Check optional arguments based on function
    if function == "power":
        if len(sys.argv) < 6:
            print("Error: The 'power' function requires the additional parameter k.")
            sys.exit(1)
        k = sys.argv[5]

    elif function == "softmax":
        if len(sys.argv) < 6:
            print("Error: The 'softmax' function requires the additional parameter dim_softmax.")
            sys.exit(1)
        dim_softmax = sys.argv[5]

    elif len(sys.argv) > 5:
        print("Warning: Extra arguments ignored.")

    roots, weights = gauss_legendre(n_points)
    roots_tan = [(mpmath.mpf(roots[i]) + 1)**2 for i in range(n_points)]
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
        f.write(f"pub global N_SOFTMAX: u32 = {dim_softmax if dim_softmax is not None else 1};\n")
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
        f.write(f"pub global COEFF_X3: Field = {int(mpmath.mpf(0.044715) * scale)};\n")
        f.write(f"pub global SQRT_2_PI: Field = {int(mpmath.sqrt(2 / mpmath.pi) * scale)};\n")
        f.write(f"pub global SQRT_PI: Field = {int(mpmath.sqrt(mpmath.pi) * scale)};\n")

    # Witness generation
    # y = f(x)

    gl_wits = []
    for i in range(int(num_inputs)):
        if function == "power":
            x_input = mpmath.mpf(mpmath.rand()) + 1
        else:
            x_input = mpmath.mpf(mpmath.rand())

        softmax_inputs = [mpmath.mpf(mpmath.rand()) for _ in range(int(dim_softmax))] if function == "softmax" else None
        
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
        elif function == "cos":
            y_input = mpmath.cos(x_input)
        elif function == "gelu":
            tanh_input = mpmath.sqrt(2 / mpmath.pi) * (x_input + 0.044715 * x_input**3)
            tanh_output = mpmath.tanh(tanh_input)
            y_input = 0.5 * x_input * (1 + tanh_output)
        elif function == "erf":
            y_input = mpmath.erf(x_input)
        elif function == "power":
            if k is None:
                print("Error: The 'power' function requires the additional parameter k.")
                sys.exit(1)
            y_input = mpmath.power(x_input, k)
        elif function == "softmax":
            if dim_softmax is None:
                print("Error: The 'softmax' function requires the additional parameter dim_softmax.")
                sys.exit(1)
            exp_values = [mpmath.exp(inp) for inp in softmax_inputs]
            sum_exp = sum(exp_values)
            softmax_outputs = [val / sum_exp for val in exp_values]
        else:
            print(f"Error: Unknown function '{function}'.")
            sys.exit(1)
        
        if function != "softmax":
            x_quantized = quantize(x_input, scale)
            y_quantized = quantize(y_input, scale)
        else:
            x_quantized = [quantize(x, scale) for x in softmax_inputs]
            y_quantized = [quantize(y, scale) for y in softmax_outputs]

        k_input = mpmath.mpf(k) if k is not None else None
        k_quantized = quantize(k_input, scale) if k_input is not None else None

        if function == "exp":
            mult_terms = []
            for r in roots:
                r = mpmath.mpf(r)
                mult_term = y_input * r
                mult_terms.append(quantize(mult_term, scale) % field_order)
            
            gl_inverses = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = (y_input - 1) / (y_input * r - r + y_input + 1)
                gl_inverses.append(quantize(gl_inverse, scale))
                
            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "denom_inverses": [str(v) for v in gl_inverses],
                    "mult_terms": [str(v) for v in mult_terms],
                }
            })

            # Compute the sum
            sum_check = mpmath.mpf(0)
            for j in range(n_points):
                w = mpmath.mpf(weights[j])
                r = mpmath.mpf(roots[j])
                sum_check += w / (r + (y_input + 1)/(y_input - 1))

            print(f"Sum check error: {mpmath.nstr(sum_check - x_input, n=15)}")


        elif function == "inv_exp":
            mult_terms = []
            for r in roots:
                r = mpmath.mpf(r)
                mult_term = y_input * r
                mult_terms.append(quantize(mult_term, scale) % field_order)

            gl_inverses = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = (1 - y_input) / ( - y_input * r + r + 1 + y_input)
                gl_inverses.append(quantize(gl_inverse, scale))

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "denom_inverses": [str(v) for v in gl_inverses],
                    "mult_terms": [str(v) for v in mult_terms],
                }
            })

            # Compute the sum
            sum_check = mpmath.mpf(0)
            for j in range(n_points):
                w = mpmath.mpf(weights[j])
                r = mpmath.mpf(roots[j])
                sum_check += w / (r + (1 + y_input)/(1 - y_input))

            print(f"Sum check error: {mpmath.nstr(sum_check - x_input, n=15)}")

        elif function == "sigmoid":
            mult_terms = []
            for r in roots:
                r = mpmath.mpf(r)
                mult_term = y_input * r
                mult_terms.append(quantize(mult_term, scale) % field_order)

            gl_inverses = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = (2 * y_input - 1) / (2 * y_input * r - r + 1)
                gl_inverses.append(quantize(gl_inverse, scale))

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "denom_inverses": [str(v) for v in gl_inverses],
                    "mult_terms": [str(v) for v in mult_terms],
                }
            })

        elif function == "tanh":
            mult_terms = []
            for r in roots:
                r = mpmath.mpf(r)
                mult_term = y_input * r
                mult_terms.append(quantize(mult_term, scale) % field_order)

            gl_inverses = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = y_input / (2 * y_input * r + 2)
                gl_inverses.append(quantize(gl_inverse, scale) % field_order)

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "denom_inverses": [str(v) for v in gl_inverses],
                    "mult_terms": [str(v) for v in mult_terms],
                }
            })

        elif function == "tan":
            y_sq = y_input ** 2

            gl_inverses = []
            mult_terms = []
            for r in roots_tan:
                r = mpmath.mpf(r)
                mult_term = y_sq * r
                mult_terms.append(quantize(mult_term, scale))
                gl_inverse = (2 * y_input) / (mult_term + 4)
                gl_inverses.append(quantize(gl_inverse, scale))

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "y_sq": str(quantize(y_sq, scale)),
                    "denom_inverses": [str(v) for v in gl_inverses],
                    "mult_terms": [str(v) for v in mult_terms]
                }
            })


        elif function == "cos":
            tan_x = mpmath.tan(x_input)
            # tan part
            tan_x_sq = tan_x ** 2

            gl_inverses = []
            mult_terms = []
            for r in roots_tan:
                r = mpmath.mpf(r)
                mult_term = tan_x_sq * r
                mult_terms.append(quantize(mult_term, scale))
                gl_inverse = (2 * tan_x) / (mult_term + 4)
                gl_inverses.append(quantize(gl_inverse, scale))

            tan_wit = {
                "y_sq": str(quantize(tan_x_sq, scale)),
                "denom_inverses": [str(v) for v in gl_inverses],
                "mult_terms": [str(v) for v in mult_terms]
            }

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "tan_x": str(quantize(tan_x, scale)),
                    "tan_witness": tan_wit,
                    "sqrt_1_p_tan2": str(quantize(mpmath.sqrt(1 + tan_x**2), scale))
                }
            })

        elif function == "gelu":
            x_sq = x_input ** 2
            x_scaled = 0.044715 * x_input
            term_2 = x_scaled * x_sq
            tanh_input = mpmath.sqrt(2 / mpmath.pi) * (x_input + 0.044715 * x_input**3)
            tanh_output = mpmath.tanh(tanh_input)

            mult_terms = []
            gl_inverses = []

            for r in roots:
                r = mpmath.mpf(r)
                mult_term = tanh_output * r
                mult_terms.append(quantize(mult_term, scale) % field_order)
            
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse = tanh_output / (2 * tanh_output * r + 2)
                gl_inverses.append(quantize(gl_inverse, scale))

            tanh_wit = {
                "denom_inverses": [str(v) for v in gl_inverses],
                "mult_terms": [str(v) for v in mult_terms],
            }

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "x_sq": str(quantize(x_sq, scale)),
                    "x_scaled": str(quantize(x_scaled, scale)),
                    "term_2": str(quantize(term_2, scale)),
                    "tanh_input": str(quantize(tanh_input, scale)),
                    "tanh_output": str(quantize(tanh_output, scale)),
                    "tanh_witness": tanh_wit
                }
            })

        elif function == "power":
            mult_terms_1 = []
            mult_terms_2 = []
            for r in roots:
                r = mpmath.mpf(r)
                mult_term_1 = y_input * r
                mult_term_2 = x_input * r
                mult_terms_1.append(quantize(mult_term_1, scale) % field_order)
                mult_terms_2.append(quantize(mult_term_2, scale) % field_order)

            gl_inverses_1 = []
            gl_inverses_2 = []
            for r in roots:
                r = mpmath.mpf(r)
                gl_inverse_1 = (1 - y_input) / (r - y_input * r + y_input + 1)
                gl_inverse_2 = k_input * (1 - x_input) / (r - x_input * r + x_input + 1)
                gl_inverses_1.append(quantize(gl_inverse_1, scale) % field_order)
                gl_inverses_2.append(quantize(gl_inverse_2, scale) % field_order)

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "denom_inverses_1": [str(v) for v in gl_inverses_1],
                    "denom_inverses_2": [str(v) for v in gl_inverses_2],
                    "mult_terms_1": [str(v) for v in mult_terms_1],
                    "mult_terms_2": [str(v) for v in mult_terms_2],
                }
            })

        elif function == "erf":
            exp_inputs = []
            exp_outputs = []
            for r in roots_tan:
                r = mpmath.mpf(r)
                exp_input = x_input**2 / 4 * r
                exp_output = mpmath.exp(-exp_input)
                exp_inputs.append(exp_input)
                exp_outputs.append(exp_output)

            x_sq_by_4 = x_input ** 2 / 4
            erf_sum = mpmath.sqrt(mpmath.pi) * y_input / x_input

            # Fill in exp_witnesses
            inv_exp_wits = []
            for i in range(n_points):
                mult_terms = []
                gl_inverses = []

                for r in roots:
                    r = mpmath.mpf(r)
                    mult_term = exp_outputs[i] * r
                    mult_terms.append(quantize(mult_term, scale) % field_order)
                    gl_inverse = (1 - exp_outputs[i]) / ( - exp_outputs[i] * r + r + 1 + exp_outputs[i])
                    gl_inverses.append(quantize(gl_inverse, scale))

                inv_exp_wits.append({
                    "denom_inverses": [str(v) for v in gl_inverses],
                    "mult_terms": [str(v) for v in mult_terms],
                })

            gl_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "inv_exp_witnesses": inv_exp_wits,
                    "inv_exp_inputs": [str(quantize(val, scale)) for val in exp_inputs],
                    "inv_exp_outputs": [str(quantize(val, scale)) for val in exp_outputs],
                    "erf_sum": str(quantize(erf_sum, scale)),
                    "x_sq_by_4": str(quantize(x_sq_by_4, scale))
                }
            })

        elif function == "softmax":
            exp_witnesses = []
            for i in range(int(dim_softmax)):
                # Create exp_witness for each dimension
                ratio = (exp_values[i] + 1) / (exp_values[i] - 1)

                gl_inverses = []
                for r in roots:
                    r = mpmath.mpf(r)
                    gl_inverse = 1 / (r + ratio)
                    gl_inverses.append(quantize(gl_inverse, scale))
                exp_witnesses.append({
                    "ratio_term": str(quantize(ratio, scale)),
                    "denom_inverses": [str(v) for v in gl_inverses]
                })

            denom_inverse = quantize(1 / sum_exp, scale)
            
            gl_wits.append({
                "inp_struct": {
                    "x": "0",
                    "y": "0",
                    "vec_x": [str(x) for x in x_quantized],
                    "vec_y": [str(y) for y in y_quantized],
                    "k": str(k_quantized) if k_quantized is not None else "0"
                },
                "wit_struct": {
                    "exp_witnesses": exp_witnesses,
                    "exp_outputs": [str(quantize(val, scale)) for val in exp_values],
                    "denom_inverse": str(denom_inverse)
                }
            })

    with open("Prover.toml", "r+") as f:
        toml_data = toml.load(f)
        toml_data["gl_wits"] = gl_wits
        f.seek(0)
        toml.dump(toml_data, f)
        f.truncate()
