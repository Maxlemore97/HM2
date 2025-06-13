import numpy as np


def f(x):
  """
  Compute the value of the function f(x1, x2).

  Args:
      x (numpy.ndarray): A numpy array of shape (2,) where x[0] = x1 and x[1] = x2.

  Returns:
      numpy.ndarray: A numpy array of shape (2,) containing the function values.
  """
  x1, x2 = x
  return np.array([2 * x1 + 4 * x2, 4 * x1 + 8 * x2 ** 3])


def jacobian(x):
  """
  Compute the Jacobian matrix Df(x).

  Args:
      x (numpy.ndarray): A numpy array of shape (2,) where x[0] = x1 and x[1] = x2.

  Returns:
      numpy.ndarray: A 2x2 numpy array representing the Jacobian matrix.
  """
  x1, x2 = x
  return np.array([[2, 4], [4, 24 * x2 ** 2]])


def newton_method(f, jacobian, x0, tol=1e-6, max_iter=100):
  """
  Perform the Newton method to solve f(x) = 0.

  Args:
      f (callable): The function f(x).
      jacobian (callable): The Jacobian matrix Df(x).
      x0 (numpy.ndarray): Initial guess for the solution.
      tol (float): Tolerance for the stopping criterion.
      max_iter (int): Maximum number of iterations.

  Returns:
      numpy.ndarray: The solution x.
      int: Number of iterations performed.
  """
  x = x0
  for i in range(max_iter):
    fx = f(x)
    dfx = jacobian(x)
    delta_x = np.linalg.solve(dfx, -fx)  # Solve Df(x) * delta_x = -f(x)
    x = x + delta_x  # Update x
    if np.linalg.norm(fx, ord=2) < tol:  # Check stopping criterion
      return x, i + 1
  raise ValueError("Newton method did not converge within the maximum number of iterations")


if __name__ == "__main__":
  # Initial guess x^(0)
  x0 = np.array([4.0, 2.0])

  # Perform Newton's method
  try:
    solution, iterations = newton_method(f, jacobian, x0)
    print("Solution found:")
    print(solution)
    print(f"Number of iterations: {iterations}")
  except ValueError as e:
    print(e)
