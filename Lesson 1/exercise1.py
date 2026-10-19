import math
import numpy as np

# --- Task 1 ---
def f30(r):
    return (2 * math.sqrt(3) / 27) * (3 - 2*r + (2/9)*(r**2)) * math.exp(-r / 3)

r_values = [
    (3/2) * (5 - math.sqrt(7)),
    (3/2) * (5 + math.sqrt(7)),
    (3/2) * (3 - math.sqrt(3)),
    (3/2) * (3 + math.sqrt(3))
]

print("--- Task 1 ---")
for r in r_values:
    print(f"r = {r:.4f} -> f30(r) = {f30(r)}")


# --- Task 2 ---
print("\n--- Task 2 ---")
for N in [5, 10, 15, 20]:
    e_approx = 0
    for n in range(N):
        e_approx += 1 / math.factorial(n)
    diff = abs(e_approx - math.e)
    print(f"N = {N}: approx = {e_approx}, diff = {diff}")


# --- Task 3 ---
def MatrixMultiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    
    C = np.zeros((rows_A, cols_B))
    for p in range(rows_A):
        for q in range(cols_B):
            for k in range(cols_A):
                C[p, q] += A[p, k] * B[k, q]
    return C

A3 = np.array([[1, 2, 3], 
               [4, 5, 6], 
               [7, 8, 9]])
B3 = np.array([[1, 0, 1], 
               [0, 2, 0], 
               [1, 0, 3]])

print("\n--- Task 3 ---")
print("function version:\n", MatrixMultiply(A3, B3))
print("python version:\n", A3 @ B3)


# --- Task 4 ---
print("\n--- Task 4 ---")
A4 = np.array([[2, 1], 
               [1, -2]])
vals4, Q4 = np.linalg.eigh(A4)

print("Eigenvalues:", vals4)
print("Eigenvectors Q:\n", Q4)
print("Norm of Q_1:", np.linalg.norm(Q4[:, 0]))
print("Norm of Q_2:", np.linalg.norm(Q4[:, 1]))
print("Dot product (orthogonality):", np.dot(Q4[:, 0], Q4[:, 1]))
print("Q^T A Q:\n", np.round(Q4.T @ A4 @ Q4, 10))


# --- Task 5A ---
print("\n--- Task 5A ---")
A5a = np.array([[-1, 0, 1], 
                [0, 2, 0], 
                [1, 0, -1]])
vals5a, Q5a = np.linalg.eigh(A5a)
Lambda5a = np.diag(vals5a)

print("Eigenvalues:", vals5a)
print("Q^T A Q (Lambda):\n", np.round(Q5a.T @ A5a @ Q5a, 10))
print("Q^T Q (Identity):\n", np.round(Q5a.T @ Q5a, 10))
print("Reconstructed A (Q Lambda Q^T):\n", np.round(Q5a @ Lambda5a @ Q5a.T, 10))
print("A^3:\n", np.linalg.matrix_power(A5a, 3))
print("Q Lambda^3 Q^T:\n", np.round(Q5a @ np.diag(vals5a**3) @ Q5a.T, 10))


# --- Task 5B ---
print("\n--- Task 5B ---")
A5b = np.array([[1, 2, 3], 
                [2, 0, 4], 
                [3, 4, -1]])
vals5b, Q5b = np.linalg.eigh(A5b)

expA = Q5b @ np.diag(np.exp(vals5b)) @ Q5b.T
exp_minusA = Q5b @ np.diag(np.exp(-vals5b)) @ Q5b.T
exp2A = Q5b @ np.diag(np.exp(2 * vals5b)) @ Q5b.T

print("exp(A):\n", expA)
print("exp(A) * exp(-A):\n", np.round(expA @ exp_minusA, 10))
print("exp(-A) * exp(A):\n", np.round(exp_minusA @ expA, 10))
print("exp(A)^2 - exp(2A) (should be 0):\n", np.round((expA @ expA) - exp2A, 10))


# --- Task 5C ---
print("\n--- Task 5C ---")
sinA = Q5b @ np.diag(np.sin(vals5b)) @ Q5b.T
cosA = Q5b @ np.diag(np.cos(vals5b)) @ Q5b.T

print("sin(A):\n", sinA)
print("cos(A):\n", cosA)
print("sin(A)^2 + cos(A)^2:\n", np.round((sinA @ sinA) + (cosA @ cosA), 10))


# --- Task 5D ---
def MatrixExpTaylor(X):
    n = len(X)
    S = np.eye(n, dtype=float)
    M = 1
    while True:
        term = np.linalg.matrix_power(X, M) / math.factorial(M)
        S_new = S + term
        
        # frobenius norm 
        diff = np.linalg.norm(S_new - S, 'fro')
        S = S_new
        
        if M > 1 and diff <= 1e-6:
            break
        M += 1
    return S

print("\n--- Task 5D ---")
expA_taylor = MatrixExpTaylor(A5b)
print("MatrixExpTaylor(A):\n", expA_taylor)