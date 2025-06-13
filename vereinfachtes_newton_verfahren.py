def f(x):
    return [2 * x[0] + 4 * x[1], 4 * x[0] + 8 * x[1]**3]

def solve_linear_system(A, b):
    # Solve 2x2 linear system A * x = b using Cramer's rule
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if det == 0:
        raise ValueError("Matrix is singular and cannot be solved")
    x0 = (b[0] * A[1][1] - b[1] * A[0][1]) / det
    x1 = (A[0][0] * b[1] - A[1][0] * b[0]) / det
    return [x0, x1]

def lu_decomposition(A, b):
    # Perform LU decomposition and solve A * x = b
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    # Decompose A into L and U
    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i, n):
            if i == j:
                L[i][i] = 1.0
            else:
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    # Forward substitution to solve L * y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # Backward substitution to solve U * x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x

def solve_with_method(A, b, method="2x2"):
    if method == "2x2":
        return solve_linear_system(A, b)
    elif method == "LU":
        return lu_decomposition(A, b)
    elif method == "LGS":
        # Directly solve using Gaussian elimination (or similar)
        return solve_linear_system(A, b)  # Placeholder for a more complex LGS solver
    else:
        raise ValueError("Unknown method")

if __name__ == "__main__":
  # Startpunkt
  x = [4.0, 2.0]

  # Jacobimatrix am Startpunkt
  J = [[2, 4], [4, 24 * x[1]**2]]

  # Iterationen
  method = "LU"  # Wähle Methode: "2x2", "LU" oder "LGS"
  for _ in range(2):  # Zwei Iterationen
      fx = f(x)  # Funktionsauswertung
      delta_x = solve_with_method(J, [-fx[0], -fx[1]], method)  # Löse J * delta_x = -f(x)
      x = [x[i] + delta_x[i] for i in range(2)]  # Aktualisiere x
      print(x)  # Gib aktualisiertes x aus