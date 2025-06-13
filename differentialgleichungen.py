def nth_order_to_first_order(coefficients, initial_conditions):
  """
  Converts an n-th order ODE into a system of first-order ODEs.

  Parameters:
  coefficients : List of coefficients [a_n, a_(n-1), ..., a_0] for the ODE:
                 a_n * y^(n) + a_(n-1) * y^(n-1) + ... + a_0 * y = 0
  initial_conditions : List of initial conditions [y(0), y'(0), ..., y^(n-1)(0)]

  Returns:
  system_of_odes : Function defining the system of first-order ODEs
  """
  n = len(coefficients) - 1  # Order of the ODE

  def system_of_odes(t, y):
    """
    Defines the system of first-order ODEs.

    Parameters:
    t : Time variable
    y : List of variables [y1, y2, ..., yn] where y1 = y, y2 = y', ..., yn = y^(n-1)

    Returns:
    dydt : List of derivatives [y1', y2', ..., yn']
    """
    dydt = [0] * n
    # First n-1 equations: y1' = y2, y2' = y3, ..., y(n-1)' = yn
    for i in range(n - 1):
      dydt[i] = y[i + 1]
    # Last equation: yn' = -(a_(n-1)/a_n)*y(n-1) - ... - (a_0/a_n)*y1
    dydt[-1] = -sum(coefficients[i] * y[i] for i in range(n)) / coefficients[-1]
    return dydt

  return system_of_odes, initial_conditions


def runge_kutta_system(f, t0, y0, t_end, n, params):
  """
  Solves a system of first-order ODEs using the 4th-order Runge-Kutta method.

  Parameters:
  f : Function defining the system of ODEs
  t0 : Initial time
  y0 : Initial values [y1(0), y2(0)]
  t_end : End time
  n : Number of steps
  params : Additional parameters for the ODE system

  Returns:
  t_values : List of time points
  y_values : List of solutions [y1, y2] at these time points
  """
  h = (t_end - t0) / n  # Step size
  t_values = [t0 + i * h for i in range(n + 1)]
  y_values = [y0]

  for i in range(n):
    t = t_values[i]
    y = y_values[-1]

    k1 = [h * dy for dy in f(t, y, *params)]
    k2 = [h * dy for dy in f(t + h / 2, [y[j] + k1[j] / 2 for j in range(len(y))], *params)]
    k3 = [h * dy for dy in f(t + h / 2, [y[j] + k2[j] / 2 for j in range(len(y))], *params)]
    k4 = [h * dy for dy in f(t + h, [y[j] + k3[j] for j in range(len(y))], *params)]

    y_next = [y[j] + (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6 for j in range(len(y))]
    y_values.append(y_next)

  return t_values, y_values


if __name__ == "__main__":
  # Parameters for the ODE: y'' + 0.3y' + 2y = 0
  coefficients = [1, 0.3, 2]  # Coefficients [a_2, a_1, a_0]
  initial_conditions = [1, 0]  # Initial conditions: y(0) = 1, y'(0) = 0

  # Convert the ODE to a system of first-order ODEs
  system_of_odes, y0 = nth_order_to_first_order(coefficients, initial_conditions)

  # Solve the system using the Runge-Kutta method
  t0 = 0
  t_end = 10
  n = 100
  t_values, y_values = runge_kutta_system(system_of_odes, t0, y0, t_end, n, ())

  # Extract y1 (displacement) and y2 (velocity)
  y1_values = [y[0] for y in y_values]
  y2_values = [y[1] for y in y_values]

  # Display results
  for t, y1, y2 in zip(t_values, y1_values, y2_values):
    print(f"{t:>10.5f} {y1:>20.5f} {y2:>20.5f}")
