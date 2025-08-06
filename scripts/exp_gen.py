import sys
import toml
from mpmath import mp, mpf, exp

mp.dps = 100  # high precision

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
    return qx, coeffs, scale, base

def generate_tables(log_base, log_scale):
    scale = 2 ** log_scale
    base = 2 ** log_base
    num_tables = log_scale // log_base

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

def write_to_constants(num_inputs, tables, log_scale, log_base, mult_ops):
    with open("src/exp_lookup/constants.nr", "w") as f:
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global LOG_S: u32 = {log_scale};\n")
        f.write(f"pub global S : Field = {scale}; // 2^{log_scale}\n")
        f.write(f"pub global LOG_BASE: u32 = {log_base};\n")
        f.write(f"pub global BASE: u32 = {base};\n")
        f.write(f"pub global MULT_OPS: [(Field, Field, Field); {len(tables) - 1}] = [\n")
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

if __name__ == "__main__":
    if len(sys.argv) == 4:
        num_inputs = int(sys.argv[1])
        log_base = int(sys.argv[2])
        log_scale = int(sys.argv[3])
    else:
        print("Usage: python exp_gen.py <num_inputs> <log_base for table size> <log_scale for quantization>")
        sys.exit(1)

    if log_scale % log_base != 0:
        print("Error: log_scale must be a multiple of log_base")
        sys.exit(1)

    exp_wits = []
    for i in range(num_inputs):
        x_input = mpf(mp.rand())
        y = exp(-x_input)

        qx, coeffs, scale, base = quantize_and_decompose(x_input, log_base, log_scale)
        tables, scale = generate_tables(log_base, log_scale)

        lookup_outputs = [tables[i][coeffs[i]] for i in range(len(coeffs))]
        data_vector, mult_ops = build_binary_tree_multiplication(lookup_outputs)

        if i == 0:
            write_to_constants(num_inputs,tables, log_scale, log_base, mult_ops)

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

