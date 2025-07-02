# Compute the expansion e^-x using Taylor series expansion
import sympy as sp

def compute_exponential(x, n_terms=20):
    """
    Compute the exponential of -x using Taylor series expansion.
    
    Parameters:
    x (float): The value to compute e^-x for.
    n_terms (int): The number of terms in the Taylor series to compute.
    
    Returns:
    float: The computed value of e^-x.
    """
    result = 0.0
    factorial = 1.0  # To hold the factorial value
    
    for n in range(n_terms):
        if n > 0:
            factorial *= n  # Update factorial for current term
        term = x ** n / factorial
        result = result + (-1) ** n * term
    
    return result

if __name__ == "__main__":
    x = 3
    y = compute_exponential(x)
    print(f"Computed e^-{x}: {y}")
    y_ac = sp.exp(-x).evalf()
    print(f"Computed e^-{x}: {y_ac}")

    x_hex = hex(int(x * 65536))
    y_hex = hex(int(y * 65536))
    print(x_hex, y_hex)