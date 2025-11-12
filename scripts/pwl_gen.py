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

def generate_pieces(func, x_start, x_end, log_num_pieces, log_scale):
    scale = 2 ** log_scale
    piece_size = (x_end - x_start) / (2 ** log_num_pieces)
    start_points = []
    linear_coeffs = []
    for i in range(2 ** log_num_pieces):
        x0 = x_start + i * piece_size
        x1 = x0 + piece_size
        y0 = func(x0)
        y1 = func(x1)
        m = (y1 - y0) / piece_size
        c = y0 - m * x0

        start_points.append(quantize(x0, scale) % field_order)
        linear_coeffs.append((quantize(m, scale) % field_order, quantize(c, scale) % field_order))

    return start_points, linear_coeffs

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python pwl_gen.py <function> <num_inputs> <x_start> <x_end> <log_num_pieces> <log_scale> <k>")
        sys.exit(1)

    func_name = sys.argv[1]
    num_inputs = int(sys.argv[2])
    x_start = float(sys.argv[3])
    x_end = float(sys.argv[4])
    log_num_pieces = int(sys.argv[5])
    log_scale = int(sys.argv[6])
    k = None

    if func_name == "power":
        if len(sys.argv) != 8:
            print("Error: power function requires 7 arguments")
            print("Usage: python pwl_gen.py power <num_inputs> <x_start> <x_end> <log_num_pieces> <log_scale> <k>")
            sys.exit(1)
        k = mpmath.mpf(sys.argv[7])

    if func_name not in function_map and func_name != "power":
        print(f"Unknown function: {func_name}")
        sys.exit(1)

    piece_size = (x_end - x_start) / (2 ** log_num_pieces)
    piece_size_quantized = quantize(piece_size, 2 ** log_scale)

    func = function_map[func_name] if func_name != "power" else mpmath.exp
    start_points, linear_coeffs = generate_pieces(func, x_start, x_end, log_num_pieces, log_scale)

    with open("src/pwl/constants.nr", "w") as f:
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global LOG_SCALE: u32 = {log_scale};\n")
        f.write(f"pub global SCALE: Field = {2 ** log_scale};\n")
        f.write(f"pub global PIECE_SIZE: Field = {piece_size_quantized};\n\n")
        f.write(f"pub global START_POINTS: [Field; {len(start_points)}] = [\n")
        f.write("    {},\n".format(", ".join(map(str, start_points))))
        f.write("];\n\n")
        f.write(f"pub global LINEAR_COEFFS: [(Field, Field); {len(linear_coeffs)}] = [\n")
        f.write("    ")
        f.write(",\n    ".join("({}, {})".format(m, b) for m, b in linear_coeffs))
        f.write("\n];\n")
        f.write(f"pub global IS_POWER: bool = {str(func_name == 'power').lower()};\n")

    pwl_wits = []
    if func_name != "power":
        for i in range(num_inputs):
            x_input = mpmath.mpf(mpmath.rand()) * (x_end - x_start) + x_start
            x_quantized = quantize(x_input, 2 ** log_scale)

            piece_index = min(int((x_input - x_start) / ((x_end - x_start) / (2 ** log_num_pieces))), (2 ** log_num_pieces) - 1)
            m, b = linear_coeffs[piece_index]

            y = func(x_input)
            y_quantized = quantize(y, 2 ** log_scale)

            pwl_wits.append({
                "x": str(x_quantized),
                "y": str(y_quantized),
                "segment_index": str(piece_index),
                "k": "0",
                "log_x": "0",
                "k_log_x": "0",
                "segment_index_2": "0"
            })

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["pwl_wits"] = pwl_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()

    else:
        for i in range(num_inputs):
            x_input = mpmath.mpf(mpmath.rand() + 1)
            y = mpmath.exp(k * mpmath.log(x_input))
            log_x = mpmath.log(x_input)
            k_log_x = k * log_x

            x_quantized = quantize(x_input, 2 ** log_scale)
            y_quantized = quantize(y, 2 ** log_scale)
            log_x_quantized = quantize(log_x, 2 ** log_scale)
            k_log_x_quantized = quantize(k_log_x, 2 ** log_scale)

            piece_index = min(int((log_x - x_start) / ((x_end - x_start) / (2 ** log_num_pieces))), (2 ** log_num_pieces) - 1)

            piece_index_2 = min(int((k_log_x - x_start) / ((x_end - x_start) / (2 ** log_num_pieces))), (2 ** log_num_pieces) - 1)

            pwl_wits.append({
                "x": str(x_quantized),
                "y": str(y_quantized),
                "segment_index": str(piece_index),
                "k": str(quantize(k, 2 ** log_scale) % field_order),
                "log_x": str(log_x_quantized % field_order),
                "k_log_x": str(k_log_x_quantized % field_order),
                "segment_index_2": str(piece_index_2)
            })

        with open("Prover.toml", "r+") as f:
            toml_data = toml.load(f)
            toml_data["pwl_wits"] = pwl_wits
            f.seek(0)
            toml.dump(toml_data, f)
            f.truncate()