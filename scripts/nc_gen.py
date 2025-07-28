import sys
import mpmath
import toml
import sympy as sp

minus_one = 21888242871839275222246405745257275088548364400416034343698204186575808495616

def newton_cotes_weights(order, prec=200):
    x = sp.symbols('x')
    points = [sp.Rational(i, order - 1) for i in range(order)]

    lagrange_polynomials = []
    for i in range(order):
        L = 1
        for j in range(order):
            if i != j:
                L *= (x - points[j]) / (points[i] - points[j])
        lagrange_polynomials.append(L)

    weights = [sp.integrate(L, (x, 0, 1)) for L in lagrange_polynomials]
    total_weight = sum(weights)
    weights = [sp.nsimplify(w / total_weight, rational=True) for w in weights]

    return weights

def quantize(value, scale):
    # return sp.nsimplify(sp.Rational(value * scale).round(), rational=True)
    return int(mpmath.nint(mpmath.mpf(value) * scale))

if __name__ == "__main__":
    if len(sys.argv) > 3:
        num_inputs = sys.argv[1]
        order = int(sys.argv[2])
        log_scale = int(sys.argv[3])
    else:
        print("Usage: python3 nc_gen.py <num_inputs> <order for NC integration approx> <log_scale for quantization>")
        sys.exit(1)

    # Use high precision
    precision_bits = 200
    sp_mp = lambda val: sp.Float(val, precision_bits)

    scale = 2 ** log_scale

    weights = newton_cotes_weights(order, prec=precision_bits)
    signs = [1 if weight >= 0 else minus_one for weight in weights]

    weights = [abs(weight) for weight in weights] # Ensure weights are non-negative
    weights = [(order - 1) * w for w in weights]  # Scale weights by (order - 1)
    quantized_weights = [quantize(w.evalf(precision_bits), scale) for w in weights]

    # Constants in src/constants.nr
    with open("src/nc_int/constants.nr", "w") as f:
        f.write(f"pub global NUM_INPUTS: u32 = {num_inputs};\n")
        f.write(f"pub global LOG_S: u32 = {log_scale};\n")
        f.write(f"pub global S : Field = {scale}; // 2^{log_scale}\n")
        f.write(f"pub global S_sq : Field = {scale * scale}; // 2^{log_scale * 2}\n")
        f.write(f"pub global NC_ORDER: u32 = {order};\n")
        f.write("pub global signs: [Field; NC_ORDER] = [\n")
        for sign in signs:
            f.write(f"    {sign},\n")
        f.write("];\n")
        f.write("pub global weights: [Field; NC_ORDER] = [\n")
        for val in quantized_weights:
            f.write(f"    {val},\n")
        f.write("];\n")

    
    # Witness generation
    # x in [0,1], y = exp(-x)
    import mpmath
    mpmath.mp.dps = 100
    
    nc_wits = []
    for i in range(int(num_inputs)):
        # Random x in [0,1]
        x = mpmath.mpf(mpmath.rand())
        y = sp_mp(sp.exp(-x))

        # Compute inverses
        lhs_inverse = quantize((1 / (1 - y)).evalf(precision_bits), scale)
        rhs_inverses = [
            quantize((1 / (i + (order - 1 - i) * y)).evalf(precision_bits), scale)
            for i in range(order)
        ]

        nc_wits.append({
            "x": str(quantize(x, scale)),
            "y": str(quantize(y, scale)),
            "lhs_inverse": str(lhs_inverse),
            "rhs_inverses": [str(v) for v in rhs_inverses]
        })

    # Witness in Prover.toml
    with open("Prover.toml", "r+") as f:
        toml_data = toml.load(f)
        toml_data["nc_wits"] = nc_wits
        f.seek(0)
        toml.dump(toml_data, f)
        f.truncate()
