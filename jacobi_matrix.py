import numpy as np


def f(x):
  """
  Compute the value of the function f(x1, x2) = x1^2 + x2^2.

  Args:
      x (numpy.ndarray): A numpy array of shape (2,) where x[0] = x1 and x[1] = x2.

  Returns:
      numpy.ndarray: A numpy array of shape (1,) containing the function value.
  """
  x1, x2 = x
  return np.array([x1 ** 2 + x2 ** 2])


def compute_jacobian(func, point, delta=1e-5):
  """
  Compute the Jacobian matrix of a function at a given point using central differences.

  Args:
      func (callable): The function for which the Jacobian is computed.
      point (numpy.ndarray): A numpy array of shape (n,) representing the evaluation point.
      delta (float): A small value for numerical differentiation.

  Returns:
      numpy.ndarray: A numpy array of shape (1, n) representing the Jacobian matrix.
  """
  num_vars = len(point)
  jacobian = np.zeros((1, num_vars))

  for i in range(num_vars):
    forward_point = point.copy()
    backward_point = point.copy()
    forward_point[i] += delta
    backward_point[i] -= delta

    jacobian[0, i] = (func(forward_point) - func(backward_point)) / (2 * delta)

  return jacobian


def linearize_function(func, point, jacobian, evaluation_point):
  """
  Linearize a function around a given point using its Jacobian matrix.

  Args:
      func (callable): The function to linearize.
      point (numpy.ndarray): A numpy array of shape (n,) representing the reference point.
      jacobian (numpy.ndarray): The Jacobian matrix at the reference point.
      evaluation_point (numpy.ndarray): A numpy array of shape (n,) representing the evaluation point.

  Returns:
      numpy.ndarray: A numpy array of shape (1,) representing the linearized function value.
  """
  func_value_at_point = func(point)
  return func_value_at_point + jacobian @ (evaluation_point - point)


if __name__ == "__main__":
  # Define the reference point (stützpunkt) x^(0) = (1, 2)^T
  reference_point = np.array([1, 2])

  # Compute the Jacobian matrix at the reference point
  jacobian_matrix = compute_jacobian(f, reference_point)

  # Print the Jacobian matrix
  print("Jacobian matrix at x^(0) = (1, 2)^T:")
  print(jacobian_matrix)

  # Linearize the function around the reference point
  evaluation_point = np.array([1.1, 2.1])
  linearized_value = linearize_function(f, reference_point, jacobian_matrix, evaluation_point)

  # Print the linearized function value
  print("Linearized function value at x = (1.1, 2.1):")
  print(linearized_value)
