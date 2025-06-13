import matplotlib.pyplot as plt


def s_stage_runge_kutta(f, t0, y0, t_end, n, s, a, b, c):
    """
    Implements the S-stage Runge-Kutta method to solve y'(t) = f(t, y).

    Parameters:
    f : Function defining the right-hand side of the differential equation f(t, y)
    t0 : Start time t0
    y0 : Initial value y(t0)
    t_end : End time t = t_end
    n : Number of steps in the interval [t0, t_end]
    s : Number of stages
    a : Coefficients matrix (s x s)
    b : Weights vector (s)
    c : Nodes vector (s)

    Returns:
    t_values : List of time points
    y_values : List of the solution y(t) at these time points
    """
    h = (t_end - t0) / n  # Step size h
    t_values = [t0 + i * h for i in range(n + 1)]  # Time points for n + 1 values
    y_values = [0] * (n + 1)  # Initialize with zeros
    y_values[0] = y0  # Initial value y(t0)

    # S-stage Runge-Kutta method
    for i in range(n):
        k = [0] * s  # Initialize stage values
        for j in range(s):
            t_stage = t_values[i] + c[j] * h
            y_stage = y_values[i] + h * sum(a[j][k_idx] * k[k_idx] for k_idx in range(j))
            k[j] = f(t_stage, y_stage)
        y_values[i + 1] = y_values[i] + h * sum(b[j] * k[j] for j in range(s))
    return t_values, y_values


# Right-hand side of the differential equation y'(t) = t^2 + 0.1y
def f(t, y):
    return t ** 2 + 0.1 * y


# Exact solution of the ODE: y(t) = -10t^2 - 200t - 2000 + 1722.5 * e^(0.05 * (2t + 3))
def exact_solution(t):
    from math import exp
    return -10 * t ** 2 - 200 * t - 2000 + 1722.5 * exp(0.05 * (2 * t + 3))


if __name__ == "__main__":
    # Task parameters
    t0 = -1.5  # Start time t = -1.5
    y0 = 0  # Initial value y(t0) = 0
    t_end = 1.5  # End time t = 1.5
    n = 5  # Number of intervals (steps)

    # S-stage Runge-Kutta parameters (classical 4th-order Runge-Kutta as an example)
    s = 4
    a = [
        [0, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, 1, 0]
    ]
    b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
    c = [0, 0.5, 0.5, 1]

    # Execute S-stage Runge-Kutta method
    t_values, y_values = s_stage_runge_kutta(f, t0, y0, t_end, n, s, a, b, c)

    # Compute exact solution (reference values)
    y_exact = [exact_solution(t) for t in t_values]

    # Compute errors (absolute difference)
    errors = [abs(y_e - y_a) for y_e, y_a in zip(y_exact, y_values)]

    # Log table data
    print(f"{'t':>10} {'y (numerical)':>20} {'y (exact)':>20} {'Error':>20}")
    for t, y_num, y_ex, err in zip(t_values, y_values, y_exact, errors):
        print(f"{t:>10.5f} {y_num:>20.5f} {y_ex:>20.5f} {err:>20.5f}")

    # Visualization of solutions
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values, 'r-o', label="S-Stage Runge-Kutta Approximation")
    plt.plot(t_values, y_exact, 'b-', label="Exact Solution")
    plt.title("S-Stage Runge-Kutta Method and Exact Solution for $y'(t) = t^2 + 0.1y$")
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Visualization of error analysis
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, errors, 'g-o', label="Absolute Errors")
    plt.title("Error Analysis of the S-Stage Runge-Kutta Method")
    plt.xlabel("$t$")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.show()