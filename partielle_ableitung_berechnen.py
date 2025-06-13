import math

# Define the function f(x, y, z)
# This function represents f(x, y, z) = x^2 * y + e^(y * z) - z
def f(vars):
  x, y, z = vars  # Unpack the variables x, y, z from the input list
  return x ** 2 * y + math.exp(y * z) - z  # Compute and return the function value


# Function to calculate the partial derivative
# This uses the central difference formula to approximate the derivative
def partial_derivative(func, var_index, point, delta=1e-5):
  point_forward = point[:]  # Create a copy of the point for forward evaluation
  point_backward = point[:]  # Create a copy of the point for backward evaluation
  point_forward[var_index] += delta  # Increment the variable at var_index by delta
  point_backward[var_index] -= delta  # Decrement the variable at var_index by delta
  # Compute the central difference: (f(x+delta) - f(x-delta)) / (2 * delta)
  return (func(point_forward) - func(point_backward)) / (2 * delta)


if __name__ == "__main__":
  # Define the point (x, y, z) = (1, 0, 2) where the derivative will be evaluated
  point = [1, 0, 2]

  # Calculate the partial derivative of f with respect to y (index 1 in the point list)
  partial_y = partial_derivative(f, 1, point)

  # Print the result of the partial derivative
  print(f"Partial derivative of f with respect to y at (1, 0, 2): {partial_y}")
