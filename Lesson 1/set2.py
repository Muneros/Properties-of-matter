from sympy import symbols, diff, integrate, exp, pi, oo, sqrt

x, y, z = symbols('x y z')
Alpha = symbols('alpha', positive=True)
r2 = x**2 + y**2 + z**2

# --- Exercise 1A ---
#exp(-alpha*(x^2+y^2+z^2)) separates into parts:
Ix = integrate(exp(-Alpha * x**2), (x, -oo, oo))
Iy = integrate(exp(-Alpha * y**2), (y, -oo, oo))
Iz = integrate(exp(-Alpha * z**2), (z, -oo, oo))
I_1a = Ix * Iy * Iz
print("Exercise 1A:", I_1a)


# --- Exercise 1B ---
# Int |G|^2 dV = N^2 * Int exp(-2*alpha*r^2) dV = 1
I_1b = integrate(exp(-2 * Alpha * r2), (x, -oo, oo), (y, -oo, oo), (z, -oo, oo))
N_1b = 1 / sqrt(I_1b)
print("Exercise 1B: N =", N_1b)


# --- Exercise 1C ---
# G = N * x * y * exp(-alpha*r^2)
I_1c = integrate((x**2) * (y**2) * exp(-2 * Alpha * r2), (x, -oo, oo), (y, -oo, oo), (z, -oo, oo))
N_1c = 1 / sqrt(I_1c)
print("Exercise 1C: N =", N_1c)


# --- Exercise 2A ---
# cc-pVDZ parameters for hydrogen (4 primitives)
alphas = [13.01, 1.962, 0.4446, 0.1220]
coeffs = [0.019685, 0.137977, 0.478148, 0.501240]

phi = 0
for c, a in zip(coeffs, alphas):
    # using normalization constant from 1B
    N_k = (2 * a / pi)**(3/4)
    phi += c * N_k * exp(-a * r2)


norm_integrand = (phi**2).expand()
I_2a = integrate(norm_integrand, (x, -oo, oo), (y, -oo, oo), (z, -oo, oo))
print("Exercise 2A (Normalization check):", float(I_2a))


# --- Exercise 2B ---
laplacian_phi = diff(phi, x, 2) + diff(phi, y, 2) + diff(phi, z, 2)
T_phi = -0.5 * laplacian_phi

kin_integrand = (phi * T_phi).expand()
I_2b = integrate(kin_integrand, (x, -oo, oo), (y, -oo, oo), (z, -oo, oo))
print("Exercise 2B (Kinetic energy <T>):", float(I_2b))


# --- Exercise 2C ---
#
r = symbols('r', positive=True)
phi_r = phi.subs(r2, r**2)

pot_integrand = (phi_r * (-1 / r) * phi_r * 4 * pi * r**2).expand()
I_2c = integrate(pot_integrand, (r, 0, oo))
print("Exercise 2C (Potential energy <V_ne>):", float(I_2c))