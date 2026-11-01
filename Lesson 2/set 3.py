import numpy as np

# --- Derivative calculation (h=1e-4 prevents 2nd-derivative precision loss) ---
def derivative(f, x, h=1e-4):
    return (f(x + h) - f(x)) / h

def f(x):
    return 4 * ((x - 2)**4)

# --- Newton method ---
def newton_method(init_guess, f, tolerance=1e-6, max_iterations=100):
    x = init_guess

    for iteration in range(max_iterations):
        # First derivative
        f_prime_x = derivative(f, x)

        # Second Derivative
        f_double_prime_x = derivative(lambda x: derivative(f, x), x)

        if abs(f_double_prime_x) < 1e-12:
            return None
        
        # --- Newton Method ---
        delta_x = -f_prime_x / f_double_prime_x

        # --- x update ---
        x += delta_x
        
        if abs(delta_x) < tolerance:
            return x

    return None

# Task 1A
print("Task 1A result:", newton_method(21, f))

# Task 1B
def f1(x):
    return 2 * x**4 + x**3 - 3 * x**2 + 5

def mult_newt(f, num_guesses, init_guess_range=[]):
    extrems = []  
    init_guesses = np.linspace(init_guess_range[0], init_guess_range[1], num_guesses)
    
    for i in range(num_guesses):
        t = newton_method(init_guesses[i], f)
        if t is not None:  
            extrems.append(t)
    
    
    return np.unique(np.round(extrems, 4))


found_extrema = mult_newt(f1, 1000, [-2, 2])
print("Task 1B extrema (x-values):", found_extrema)


for x_val in found_extrema:
    second_deriv = derivative(lambda x: derivative(f1, x), x_val)
    kind = "Local Minimum" if second_deriv > 0 else "Local Maximum"
    print(f"x = {x_val:7.4f}, f(x) = {f1(x_val):7.4f} -> {kind}")