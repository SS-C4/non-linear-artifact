import sys
import toml
import sympy as sp
import mpmath

mpmath.mp.dps = 100
field_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def quantize(value, scale):
    return int(mpmath.nint(mpmath.mpf(value) * scale))

def gauss_legendre(n):
    x = sp.symbols('x')
    Pn = sp.legendre(n, x)
    roots = sp.solve(Pn, x)
    Pn_prime = sp.diff(Pn, x)

    weights = []
    for xi in roots:
        xi_numeric = xi.evalf(mpmath.mp.dps)
        dPn = Pn_prime.subs(x, xi_numeric).evalf(mpmath.mp.dps)
        wi = 2 / ((1 - xi_numeric**2) * dPn**2)
        weights.append(mpmath.nstr(wi, n=100))

    roots = [mpmath.nstr(xi.evalf(mpmath.mp.dps), n=100) for xi in roots]
    
    return roots, weights

if __name__ == "__main__":
    if len(sys.argv) > 3:
        x_input = sys.argv[1]
        n_points = int(sys.argv[2])
        log_scale = int(sys.argv[3])
    else:
        print("Usage: python3 gl_gen.py <input> <number of points for GL quadrature> <log_scale for quantization>")
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
    # y = exp(-x)
    x_val = mpmath.mpf(x_input)
    y_val = mpmath.exp(-x_val)
    x_quantized = quantize(x_val, scale)
    y_quantized = quantize(y_val, scale)

    ratio = (1 + y_val) / (1 - y_val)

    gl_inverses = []
    for r in roots:
        r = mpmath.mpf(r)
        gl_inverse = 1 / (r + ratio)
        gl_inverses.append(quantize(gl_inverse, scale))

    # Witness in Prover.toml
    with open("Prover.toml", "r+") as f:
        toml_data = toml.load(f)
        toml_data["gl_wit"] = {
            "x": str(x_quantized),
            "y": str(y_quantized),
            "ratio": str(quantize(ratio, scale)),
            "gl_inverses": [str(v) for v in gl_inverses]
        }
        f.seek(0)
        toml.dump(toml_data, f)
        f.truncate()

