def f(x):
    # Function f(x)
    return [2 * x[0] + 4 * x[1], 4 * x[0] + 8 * x[1]**3]

def jacobian(x):
    # Jacobian matrix Df(x)
    return [[2, 4], [4, 24 * x[1]**2]]

def solve_linear_system(A, b):
    # Solve 2x2 linear system A * x = b using Cramer's rule
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if det == 0:
        raise ValueError("Matrix is singular and cannot be solved")
    x0 = (b[0] * A[1][1] - b[1] * A[0][1]) / det
    x1 = (A[0][0] * b[1] - A[1][0] * b[0]) / det
    return [x0, x1]

def damped_newton(f, jacobian, x0, tol=1e-6, k_max=4):
    x = x0
    for k in range(k_max):
        fx = f(x)  # Evaluate f(x)
        J = jacobian(x)  # Evaluate Jacobian Df(x)
        delta_x = solve_linear_system(J, [-fx[0], -fx[1]])  # Solve J * delta_x = -f(x)

        # Damping factor
        lambda_ = 1.0
        while True:
            x_new = [x[i] + lambda_ * delta_x[i] for i in range(2)]  # Update x with damping
            if sum(abs(f(x_new)[i]) for i in range(2)) < sum(abs(f(x)[i]) for i in range(2)):
                break
            lambda_ /= 2  # Reduce damping factor

        x = x_new  # Update x
        if sum(abs(fx[i]) for i in range(2)) < tol:  # Check convergence
            return x, k + 1
    raise ValueError("Damped Newton method did not converge within the maximum number of iterations")

if __name__ == "__main__":
    # Initial guess
    x0 = [4.0, 2.0]
    tol = 1e-6
    k_max = 4

    # Solve using damped Newton method
    try:
        solution, iterations = damped_newton(f, jacobian, x0, tol, k_max)
        print("Solution found:")
        print(solution)
        print(f"Number of iterations: {iterations}")
    except ValueError as e:
        print(e)