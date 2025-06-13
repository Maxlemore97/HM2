import matplotlib.pyplot as plt


def runge_kutta_method(f, t0, y0, t_end, n):
  """
  Implements the 4th-order Runge-Kutta method to solve y'(t) = f(t, y).

  Parameters:
  f : Function defining the right-hand side of the differential equation f(t, y)
  t0 : Start time t0
  y0 : Initial value y(t0)
  t_end : End time t = t_end
  n : Number of steps in the interval [t0, t_end]

  Returns:
  t_values : List of time points
  y_values : List of the solution y(t) at these time points
  """
  h = (t_end - t0) / n  # Step size h
  t_values = [t0 + i * h for i in range(n + 1)]  # Time points for n + 1 values
  y_values = [0] * (n + 1)  # Initialize with zeros
  y_values[0] = y0  # Initial value y(t0)

  # Runge-Kutta 4th order method for all time points
  for i in range(n):
    k1 = h * f(t_values[i], y_values[i])
    k2 = h * f(t_values[i] + h / 2, y_values[i] + k1 / 2)
    k3 = h * f(t_values[i] + h / 2, y_values[i] + k2 / 2)
    k4 = h * f(t_values[i] + h, y_values[i] + k3)
    y_values[i + 1] = y_values[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
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

  # Execute Runge-Kutta method
  t_values, y_values = runge_kutta_method(f, t0, y0, t_end, n)

  # Compute exact solution (reference values)
  y_exact = [exact_solution(t) for t in t_values]

  # Compute errors (absolute difference)
  errors = [abs(y_e - y_a) for y_e, y_a in zip(y_exact, y_values)]

  # Visualization of solutions
  plt.figure(figsize=(10, 6))
  plt.plot(t_values, y_values, 'r-o', label="Runge-Kutta Approximation")
  plt.plot(t_values, y_exact, 'b-', label="Exact Solution")
  plt.title("Runge-Kutta Method and Exact Solution for $y'(t) = t^2 + 0.1y$")
  plt.xlabel("$t$")
  plt.ylabel("$y$")
  plt.grid(True)
  plt.legend()
  plt.show()

  # Visualization of error analysis
  plt.figure(figsize=(10, 6))
  plt.plot(t_values, errors, 'g-o', label="Absolute Errors")
  plt.title("Error Analysis of the Runge-Kutta Method")
  plt.xlabel("$t$")
  plt.ylabel("Error")
  plt.grid(True)
  plt.legend()
  plt.show()
