import sys
import toml
import mpmath

mpmath.mp.dps = 100
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

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
    if len(sys.argv) > 3:
        num_inputs = sys.argv[1]
        n_points = int(sys.argv[2])
        log_scale = int(sys.argv[3])
    else:
        print("Usage: python3 gl_gen.py <num_inputs> <number of points for GL quadrature> <log_scale for quantization>")
        sys.exit(1)

    roots, weights = gauss_legendre(n_points)
    scale = 2 ** log_scale

    # Quantize roots and weights
    quantized_weights = [quantize(w, scale) for w in weights]
    quantized_roots = [quantize(r, scale) for r in roots]

    # If negative, map to field
    quantized_weights = [str(int(mpmath.mpf(w)) % field_order) for w in quantized_weights]
    quantized_roots = [str(int(mpmath.mpf(r)) % field_order) for r in quantized_roots]

    with open("src/gl_quad/constants.nr", "w") as f:
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global LOG_S: u32 = {log_scale};\n")
        f.write(f"pub global S: Field = {scale};\n")
        f.write(f"pub global S_sq: Field = {scale * scale};\n")
        f.write(f"pub global N_POINTS: u32 = {n_points};\n")
        f.write("pub global GL_ROOTS: [Field; N_POINTS] = [\n")
        for r in quantized_roots:
            f.write(f"    {int(mpmath.mpf(r))},\n")
        f.write("];\n")
        f.write("pub global GL_WEIGHTS: [Field; N_POINTS] = [\n")
        for w in quantized_weights:
            f.write(f"    {int(mpmath.mpf(w))},\n")
        f.write("];\n")

    # Witness generation
    # y = exp(-x) for random x in [0,1] for num_inputs
    gl_wits = []
    for i in range(int(num_inputs)):
        x_input = mpmath.mpf(mpmath.rand())
        y_input = mpmath.exp(-x_input)
        x_quantized = quantize(x_input, scale)
        y_quantized = quantize(y_input, scale)

        ratio = (1 + y_input) / (1 - y_input)

        gl_inverses = []
        for r in roots:
            r = mpmath.mpf(r)
            gl_inverse = 1 / (r + ratio)
            gl_inverses.append(quantize(gl_inverse, scale))

        gl_wits.append({
            "x": str(x_quantized),
            "y": str(y_quantized),
            "ratio": str(quantize(ratio, scale)),
            "gl_inverses": [str(v) for v in gl_inverses]
        })

    with open("Prover.toml", "r+") as f:
        toml_data = toml.load(f)
        toml_data["gl_wits"] = gl_wits
        f.seek(0)
        toml.dump(toml_data, f)
        f.truncate()
