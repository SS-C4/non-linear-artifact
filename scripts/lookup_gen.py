import sys
import toml
from mpmath import mp, mpf, exp, sqrt, pi

mp.dps = 100  # high precision
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def quantize_and_decompose(x_str, log_base, log_scale):
    x = mpf(x_str)
    scale = 2 ** log_scale
    base = 2 ** log_base

    qx = int(mp.nint(x * scale))
    coeffs = []
    temp = qx
    while temp > 0:
        coeffs.append(temp % base)
        temp //= base

    # Ensure we have len(coeffs) == log_scale // log_base by padding with zeros
    while len(coeffs) < log_scale // log_base:
        coeffs.append(0)
    return qx, coeffs, scale, base

def generate_tables(log_base, log_scale, is_softmax=False):
    scale = 2 ** log_scale
    base = 2 ** log_base
    num_tables = log_scale // log_base + (1 if is_softmax else 0)

    tables = []
    for i in range(num_tables):
        b_to_i = base ** i
        row = []
        for j in range(base):
            val = exp(-mpf(j * b_to_i) / scale)
            qval = int(mp.nint(val * scale))
            row.append(qval)
        tables.append(row)
    return tables, scale

def generate_table(func, log_scale):
    scale = 2 ** log_scale
    table = []
    for j in range(scale):
        val = func(mpf(j) / scale)
        qval = int(mp.nint(val * scale))
        table.append(qval)
    return table, scale

def evaluate_lookup(coeffs, tables, scale):
    product = 1
    for i, ai in enumerate(coeffs):
        product *= tables[i][ai]
    return mpf(product) / scale**len(coeffs)

def build_binary_tree_multiplication(input_values):
    n = len(input_values)
    total_size = 2 * n - 1
    data_vector = [None] * total_size
    data_vector[:n] = input_values  # Fill in the initial values

    operations = []
    next_free = n  # Start filling results from index n

    def build(indices):
        nonlocal next_free
        if len(indices) == 1:
            return indices[0]
        else:
            mid = len(indices) // 2
            left = build(indices[:mid])
            right = build(indices[mid:])
            result_index = next_free
            next_free += 1

            operations.append((left, right, result_index))
            data_vector[result_index] = data_vector[left] * data_vector[right]
            # Truncate the bottom scale bits
            data_vector[result_index] = data_vector[result_index] >> log_scale

            return result_index

    # Start building from indices of original inputs
    build(list(range(n)))

    return data_vector, operations

def write_to_constants(func_type, num_inputs, softmax_size, tables, custom_table, log_scale, log_base, mult_ops):
    with open("src/lookup/constants.nr", "w") as f:
        f.write(f"pub type FUNC_TYPE = super::structs::{func_type};\n")
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global NUM_SOFTMAX: u32 = {softmax_size};\n")
        f.write(f"pub global LOG_S: u32 = {log_scale};\n")
        f.write(f"pub global S : Field = {scale}; // 2^{log_scale}\n")
        f.write(f"pub global LOG_BASE: u32 = {log_base};\n")
        f.write(f"pub global BASE: u32 = {base};\n")
        f.write(f"pub global MULT_OPS: [(Field, Field, Field); {len(mult_ops)}] = [\n")
        for i in range(len(mult_ops)):
            f.write("    (")
            f.write(", ".join(str(x) for x in mult_ops[i]))
            f.write("),\n")
        f.write("];\n")
        f.write(f"pub global NUM_TABLES: u32 = {len(tables)};\n")
        f.write(f"pub global BASE_POWERS: [Field; {len(tables)}] = [\n")
        for i in range(len(tables)):
            f.write(f"    {base ** i},\n")
        f.write("];\n")
        f.write("pub global EXP_TABLES: [[Field; {}]; {}] = [\n".format(base, len(tables)))
        for table in tables:
            f.write("    [{}],\n".format(", ".join(map(str, table))))
        f.write("];\n")
        if len(custom_table) > 0:
            f.write("pub global CUSTOM_TABLE: [Field; {}] = [\n".format(len(custom_table)))
            f.write("    {},\n".format(", ".join(map(str, custom_table))))
            f.write("];\n")
        else:
            f.write("pub global CUSTOM_TABLE: [Field; 0] = [];\n")
        f.write(f"pub global COEFF_X3: Field = {int(mpf(0.044715) * scale)};\n")
        f.write(f"pub global SQRT_2_PI: Field = {int(sqrt(2 / pi) * scale)};\n")

function_map = {
    "inv_exp": "InvExpLookupWitness",
    "sigmoid": "SigmoidLookupWitness",
    "gelu": "GeluLookupWitness",
    "erf": "ErfLookupWitness",
    "power": "PowerLookupWitness",
    "tanh": "TanhLookupWitness",
    "cos": "CosineLookupWitness",
    "tan": "TanLookupWitness",
    "softmax": "SoftMaxLookupWitness"
}

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 gl_gen.py <function> <num_inputs> <n_points> <log_scale> [<k> if function is 'power'] [<dim_softmax> if function is 'softmax']")
        print("\nArguments:")
        print("  <function>        : 'exp', 'inv_exp', 'sigmoid', 'tanh', 'tan', 'cos', 'power', 'softmax'.")
        print("  <num_inputs>      : Number of inputs required for the function.")
        print("  <log_base>       : Log base for quantization (integer).")
        print("  <log_scale>       : Log scale for quantization (integer).")
        print("  <k>               : Required if function is 'power'.")
        print("  <dim_softmax>     : Required if function is 'softmax'.")
        sys.exit(1)

    # Parse required arguments
    func = sys.argv[1]
    num_inputs = int(sys.argv[2])
    log_base = int(sys.argv[3])
    log_scale = int(sys.argv[4])

    k = None
    softmax_size = None

    # Check optional arguments based on function
    if func == "power":
        if len(sys.argv) < 6:
            print("Error: The 'power' function requires the additional parameter k.")
            sys.exit(1)
        k = sys.argv[5]
        k_quantized = int(mp.nint(mpf(k) * (2 ** log_scale)))

    elif func == "softmax":
        if len(sys.argv) < 6:
            print("Error: The 'softmax' function requires the additional parameter softmax_size.")
            sys.exit(1)
        softmax_size = int(sys.argv[5])

    elif len(sys.argv) > 5:
        print("Warning: Extra arguments ignored.")

    if func == "cos":
        assert log_base == log_scale, "Cannot decompose table"

    if log_scale % log_base != 0:
        print("Error: log_scale must be a multiple of log_base")
        sys.exit(1)

    if func == "inv_exp":
        exp_wits = []
        tables, scale = generate_tables(log_base, log_scale, is_softmax=False)

        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            y = exp(-x_input)

            qx, coeffs, scale, base = quantize_and_decompose(x_input, log_base, log_scale)

            lookup_outputs = [tables[i][coeffs[i]] for i in range(len(coeffs))]
            data_vector, mult_ops = build_binary_tree_multiplication(lookup_outputs)

            if i == 0:
                softmax_size = 1  # Not used for inv_exp
                write_to_constants(function_map[func], num_inputs, softmax_size, tables, [], log_scale, log_base, mult_ops)

            x_quantized = int(mp.nint(mpf(x_input) * scale))
            y_quantized = int(mp.nint(y * scale))

            exp_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k) if k is not None else "0",
                },
                "wit_struct": {
                    "x_decomp": [str(c) for c in coeffs],
                    "lookup_mults": [str(v) for v in data_vector],
                }
            })

        # Witness in Prover.toml
        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = exp_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "sigmoid":
        sigmoid_wits = []
        tables, scale = generate_tables(log_base, log_scale, is_softmax=False)

        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            exp_output = exp(-x_input)
            y = 1 / (1 + exp(-x_input))

            qx, coeffs, scale, base = quantize_and_decompose(x_input, log_base, log_scale)

            lookup_outputs = [tables[i][coeffs[i]] for i in range(len(coeffs))]
            data_vector, mult_ops = build_binary_tree_multiplication(lookup_outputs)

            if i == 0:
                softmax_size = 1  # Not used for sigmoid
                write_to_constants(function_map[func], num_inputs, softmax_size, tables, [], log_scale, log_base, mult_ops)

            x_quantized = int(mp.nint(mpf(x_input) * scale))
            y_quantized = int(mp.nint(y * scale))

            sigmoid_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": "0",
                },
                "wit_struct": {
                    "exp_output": str(int(mp.nint(exp_output * scale))),
                    "inv_exp_wit": {
                        "x_decomp": [str(c) for c in coeffs],
                        "lookup_mults": [str(v) for v in data_vector],
                    },
                }
            })

        # Witness in Prover.toml
        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = sigmoid_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "softmax":
        softmax_wits = []
        tables, scale = generate_tables(log_base, log_scale, is_softmax=True)

        for i in range(num_inputs):
            vec_x = [mpf(mp.rand()) for _ in range(softmax_size)]
            exp_vals = [exp(x) for x in vec_x]
            sum_exp = sum(exp_vals)
            shift = mp.log(sum_exp)
            vec_y = [ev / sum_exp for ev in exp_vals]

            shift_quantized = int(mp.nint(shift * scale))
            vec_x_quantized = [int(mp.nint(mpf(x) * scale)) for x in vec_x]
            vec_y_quantized = [int(mp.nint(mpf(y) * scale)) for y in vec_y]

            exp_wits = []
            for j in range(softmax_size):
                x_quantized = shift_quantized - vec_x_quantized[j]
                x_input = mpf(x_quantized) / scale
                y = vec_y[j]

                qx, coeffs, scale, base = quantize_and_decompose(x_input, log_base, log_scale)

                lookup_outputs = [tables[i][coeffs[i]] for i in range(len(coeffs))]
                data_vector, mult_ops = build_binary_tree_multiplication(lookup_outputs)

                if i == 0 and j == 0:
                    write_to_constants(function_map[func], num_inputs, softmax_size, tables, [], log_scale, log_base, mult_ops)

                # x_quantized = int(mp.nint(mpf(x_input) * scale))
                y_quantized = int(mp.nint(y * scale))

                exp_wits.append({
                    "inp_struct": {
                        "x": str(x_quantized),
                        "y": str(y_quantized),
                        "vec_x": [str(xq) for xq in vec_x_quantized],
                        "vec_y": [str(yq) for yq in vec_y_quantized],
                        "k": "0",
                    },
                    "wit_struct": {
                        "x_decomp": [str(c) for c in coeffs],
                        "lookup_mults": [str(v) for v in data_vector],
                    }
                })

            shift_quantized = int(mp.nint(shift * scale))
            softmax_wits.append({
                "inp_struct": {
                    "x": str(0),
                    "y": str(0),
                    "vec_x": [str(xq) for xq in vec_x_quantized],
                    "vec_y": [str(yq) for yq in vec_y_quantized],
                    "k": "0",
                },
                "wit_struct": {
                    "exp_wits": exp_wits,
                    "shift": str(shift_quantized),
                }
            })

        # Witness in Prover.toml
        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = softmax_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "cos":
        cosine_wits = []
        table, scale = generate_table(mp.cos, log_scale)

        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            y = table[int(mp.nint(x_input * scale))] / scale
            x_quantized = int(mp.nint(x_input * scale))
            y_quantized = int(mp.nint(y * scale))

            cosine_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": "0",
                },
                "wit_struct": {
                    "_dummy": "0",
                }
            })

            if i == 0:
                softmax_size = 1  # Not used for cosine
                base = 2 ** log_base
                write_to_constants(function_map[func], num_inputs, softmax_size, [], table, log_scale, log_base, [(0,0,0)])

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = cosine_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "tan":
        tangent_wits = []
        table, scale = generate_table(mp.tan, log_scale)
        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            y = table[int(mp.nint(x_input * scale))] / scale
            x_quantized = int(mp.nint(x_input * scale))
            y_quantized = int(mp.nint(y * scale))

            tangent_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": "0",
                },
                "wit_struct": {
                    "_dummy": "0",
                }
            })

            if i == 0:
                softmax_size = 1  # Not used for tangent
                base = 2 ** log_base
                write_to_constants(function_map[func], num_inputs, softmax_size, [], table, log_scale, log_base, [(0,0,0)])

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = tangent_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "power":
        power_wits = []
        tables, scale = generate_tables(log_base, log_scale, is_softmax=False)

        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            y = exp(k * mp.log(x_input))
            x_quantized = int(mp.nint(mpf(x_input) * scale))
            y_quantized = int(mp.nint(mpf(y) * scale))

            log_x = mp.log(x_input)
            k_log_x = k * log_x

            # x = exp(log_x)
            qx, coeffs, scale, base = quantize_and_decompose(log_x, log_base, log_scale)
            lookup_outputs = [tables[i][coeffs[i]] for i in range(len(coeffs))]
            data_vector, mult_ops = build_binary_tree_multiplication(lookup_outputs)

            inv_exp_wit_x = {
                "x_decomp": [str(c) for c in coeffs],
                "lookup_mults": [str(v) for v in data_vector],
            }

            # y = exp(k * log_x)
            qx, coeffs, scale, base = quantize_and_decompose(k_log_x, log_base, log_scale)
            lookup_outputs = [tables[i][coeffs[i]] for i in range(len(coeffs))]
            data_vector, mult_ops = build_binary_tree_multiplication(lookup_outputs)

            inv_exp_wit_y = {
                "x_decomp": [str(c) for c in coeffs],
                "lookup_mults": [str(v) for v in data_vector],
            }

            if i == 0 and j == 0:
                write_to_constants(function_map[func], num_inputs, softmax_size, tables, [], log_scale, log_base, mult_ops)

            power_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": str(k_quantized),
                },
                "wit_struct": {
                    "inv_exp_wit_x": inv_exp_wit_x,
                    "inv_exp_wit_y": inv_exp_wit_y,
                    "log_x": str(int(mp.nint(log_x * scale))),
                    "k_log_x": str(int(mp.nint(k_log_x * scale))),
                }
            })

        # Witness in Prover.toml
        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = power_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "tanh":
        tanh_wits = []
        tables, scale = generate_tables(log_base, log_scale, is_softmax=False)

        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            y = mp.tanh(x_input)
            exp_output = exp(- x_input)
            out_sq = exp_output * exp_output

            qx, coeffs, scale, base = quantize_and_decompose(x_input, log_base, log_scale)

            lookup_outputs = [tables[i][coeffs[i]] for i in range(len(coeffs))]
            data_vector, mult_ops = build_binary_tree_multiplication(lookup_outputs)

            if i == 0:
                softmax_size = 1  # Not used for tanh
                write_to_constants(function_map[func], num_inputs, softmax_size, tables, [], log_scale, log_base, mult_ops)

            x_quantized = int(mp.nint(mpf(x_input) * scale))
            y_quantized = int(mp.nint(mpf(y) * scale))

            tanh_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": "0",
                },
                "wit_struct": {
                    "exp_output": str(int(mp.nint(exp_output * scale))),
                    "out_sq": str(int(mp.nint(out_sq * scale))),
                    "inv_exp_witness": {
                        "x_decomp": [str(c) for c in coeffs],
                        "lookup_mults": [str(v) for v in data_vector],
                    }
                }
            })

        # Witness in Prover.toml
        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = tanh_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "gelu":
        gelu_wits = []
        tables, scale = generate_tables(log_base, log_scale, is_softmax=False)

        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            x_sq = x_input * x_input
            x_scaled = x_input * 0.044715
            term_2 = x_scaled * x_sq
            tanh_input = sqrt(2 / pi) * (x_input + 0.044715 * x_input**3)
            exp_output = exp(- tanh_input)
            out_sq = exp_output * exp_output
            tanh_output = mp.tanh(tanh_input)
            y = 0.5 * x_input * (1 + tanh_output)

            x_quantized = int(mp.nint(mpf(x_input) * scale))
            y_quantized = int(mp.nint(mpf(y) * scale))

            qx, coeffs, scale, base = quantize_and_decompose(tanh_input, log_base, log_scale)
            lookup_outputs = [tables[i][coeffs[i]] for i in range(len(coeffs))]
            data_vector, mult_ops = build_binary_tree_multiplication(lookup_outputs)

            if i == 0:
                softmax_size = 1  # Not used for gelu
                write_to_constants(function_map[func], num_inputs, softmax_size, tables, [], log_scale, log_base, mult_ops)

            gelu_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": "0",
                },
                "wit_struct": {
                    "x_sq": str(int(mp.nint(x_sq * scale))),
                    "x_scaled": str(int(mp.nint(x_scaled * scale))),
                    "term_2": str(int(mp.nint(term_2 * scale))),
                    "tanh_input": str(int(mp.nint(tanh_input * scale))),
                    "tanh_output": str(int(mp.nint(tanh_output * scale))),
                    "tanh_lookup_witness": {
                        "exp_output": str(int(mp.nint(exp_output * scale))),
                        "out_sq": str(int(mp.nint(out_sq * scale))),
                        "inv_exp_witness": {
                            "x_decomp": [str(c) for c in coeffs],
                            "lookup_mults": [str(v) for v in data_vector],
                        }
                    }
                }
            })

        # Witness in Prover.toml
        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = gelu_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "erf":
        erf_wits = []
        table, scale = generate_table(mp.erf, log_scale)

        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            y = table[int(mp.nint(x_input * scale))] / scale
            x_quantized = int(mp.nint(x_input * scale))
            y_quantized = int(mp.nint(y * scale))

            erf_wits.append({
                "inp_struct": {
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "vec_x": [str(x_quantized)],
                    "vec_y": [str(y_quantized)],
                    "k": "0",
                },
                "wit_struct": {
                    "_dummy": "0",
                }
            })

            if i == 0:
                softmax_size = 1  # Not used for erf
                base = 2 ** log_base
                write_to_constants(function_map[func], num_inputs, softmax_size, [], table, log_scale, log_base, [(0,0,0)])

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["lookup_wits"] = erf_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

        
