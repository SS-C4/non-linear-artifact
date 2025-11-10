import sys
import toml
from mpmath import mp, mpf, exp

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

def generate_cosine_table(log_scale):
    scale = 2 ** log_scale
    table = []
    for j in range(scale):
        val = mp.cos(mpf(j) / scale)
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

def write_to_constants(num_inputs, softmax_size, tables, cosine_table, log_scale, log_base, mult_ops):
    with open("src/exp_lookup/constants.nr", "w") as f:
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
        f.write("pub global COSINE_TABLE: [Field; {}] = [\n".format(len(cosine_table)))
        f.write("    {},\n".format(", ".join(map(str, cosine_table))))
        f.write("];\n")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python exp_gen.py <func> <num_inputs> <log_base> <log_scale> [<softmax_size> if func is 'softmax']")
        sys.exit(1)

    func = sys.argv[1]

    try:
        num_inputs = int(sys.argv[2])
        log_base = int(sys.argv[3])
        log_scale = int(sys.argv[4])
    except ValueError:
        print("Error: <num_inputs>, <log_base>, and <log_scale> must be integers")
        sys.exit(1)

    softmax_size = None
    if func == "softmax":
        if len(sys.argv) != 6:
            print("Error: 'softmax' function requires <softmax_size> argument.")
            sys.exit(1)
        try:
            softmax_size = int(sys.argv[5])
        except ValueError:
            print("Error: <softmax_size> must be an integer")
            sys.exit(1)
    else:
        if len(sys.argv) != 5:
            print("Error: Only 'softmax' takes a <softmax_size> argument.")
            sys.exit(1)

    if func == "cosine":
        assert log_base == log_scale, "For cosine, log_base must equal log_scale"

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
                write_to_constants(num_inputs, softmax_size, tables, [], log_scale, log_base, mult_ops)

            x_quantized = int(mp.nint(mpf(x_input) * scale))
            y_quantized = int(mp.nint(y * scale))

            exp_wits.append({
                "x": str(x_quantized),
                "y": str(y_quantized),
                "x_decomp": [str(c) for c in coeffs],
                "lookup_mults": [str(v) for v in data_vector],
            })

        # Witness in Prover.toml
        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["_exp_wits"] = exp_wits
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
                    write_to_constants(num_inputs, softmax_size, tables, [], log_scale, log_base, mult_ops)

                # x_quantized = int(mp.nint(mpf(x_input) * scale))
                y_quantized = int(mp.nint(y * scale))

                exp_wits.append({
                    "x": str(x_quantized),
                    "y": str(y_quantized),
                    "x_decomp": [str(c) for c in coeffs],
                    "lookup_mults": [str(v) for v in data_vector],
                })

            shift_quantized = int(mp.nint(shift * scale))
            softmax_wits.append({
                "vec_x": [str(xq) for xq in vec_x_quantized],
                "vec_y": [str(yq) for yq in vec_y_quantized],
                "exp_wits": exp_wits,
                "shift": str(shift_quantized),
            })

        # Witness in Prover.toml
        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["_softmax_wits"] = softmax_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    elif func == "cosine":
        cosine_wits = []
        table, scale = generate_cosine_table(log_scale)

        for i in range(num_inputs):
            x_input = mpf(mp.rand())
            y = table[int(mp.nint(x_input * scale))] / scale
            x_quantized = int(mp.nint(x_input * scale))
            y_quantized = int(mp.nint(y * scale))

            cosine_wits.append({
                "x": str(x_quantized),
                "y": str(y_quantized),
            })

            if i == 0:
                softmax_size = 1  # Not used for cosine
                base = 2 ** log_base
                write_to_constants(num_inputs, softmax_size, [], table, log_scale, log_base, [(0,0,0)])

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["_cosine_wits"] = cosine_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()


