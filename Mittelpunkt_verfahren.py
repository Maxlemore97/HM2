import numpy as np
import matplotlib.pyplot as plt


def midpoint_method(f, t0, y0, t_end, n):
    """
    Implements the midpoint method to solve y'(t) = f(t, y).

    Parameters:
    f : Function defining the right-hand side of the differential equation f(t, y)
    t0 : Start time t0
    y0 : Initial value y(t0)
    t_end : End time t = t_end
    n : Number of steps in the interval [t0, t_end]

    Returns:
    t_values : Array of time points
    y_values : Array of the solution y(t) at these time points
    """
    h = (t_end - t0) / n  # Step size h
    t_values = np.linspace(t0, t_end, n + 1)  # Time points for n + 1 values
    y_values = np.zeros(n + 1)  # Initialize with zeros
    y_values[0] = y0  # Initial value y(t0)

    # Midpoint method for all time points
    for i in range(n):
        t_mid = t_values[i] + h / 2
        y_mid = y_values[i] + (h / 2) * f(t_values[i], y_values[i])
        y_values[i + 1] = y_values[i] + h * f(t_mid, y_mid)
    return t_values, y_values


# Right-hand side of the differential equation y'(t) = t^2 + 0.1y
def f(t, y):
    return t ** 2 + 0.1 * y


# Exact solution of the ODE: y(t) = -10t^2 - 200t - 2000 + 1722.5 * e^(0.05 * (2t + 3))
def exact_solution(t):
    return -10 * t ** 2 - 200 * t - 2000 + 1722.5 * np.exp(0.05 * (2 * t + 3))


if __name__ == "__main__":
    # Task parameters
    t0 = -1.5  # Start time t = -1.5
    y0 = 0  # Initial value y(t0) = 0
    t_end = 1.5  # End time t = 1.5
    n = 5  # Number of intervals (steps)

    # Execute midpoint method
    t_values, y_values = midpoint_method(f, t0, y0, t_end, n)

    # Compute exact solution (reference values)
    y_exact = exact_solution(t_values)

    # Compute errors (absolute difference)
    errors = np.abs(y_exact - y_values)

    # Display results in the console
    print("t-values:", t_values)
    print("y-values (Midpoint, numerical):", y_values)
    print("y-values (Exact solution):", y_exact)
    print("Errors:", errors)

    # Visualization of solutions
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_values, 'r-o', label="Midpoint Approximation")
    plt.plot(t_values, y_exact, 'b-', label="Exact Solution")
    plt.title("Midpoint Method and Exact Solution for $y'(t) = t^2 + 0.1y$")
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Visualization of error analysis
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, errors, 'g-o', label="Absolute Errors")
    plt.title("Error Analysis of the Midpoint Method")
    plt.xlabel("$t$")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.show()