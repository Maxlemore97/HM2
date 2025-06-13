# ------------------------------------------------------------
#  Gauss-Newton & gedämpftes Gauss-Newton (NumPy/SymPy)
# ------------------------------------------------------------
import numpy as np
import sympy as sp


# ---------- 1. ALGEBRAISCHE HILFSFUNKTIONEN -----------------
def solve_2x2(matrix, vector):
  """
    Löst ein 2x2-Gleichungssystem exakt.

    Parameter:
    matrix: np.array, 2x2-Koeffizientenmatrix
    vector: np.array, rechte Seite des Gleichungssystems

    Rückgabe:
    np.array: Lösung des Gleichungssystems
    """
  determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
  if abs(determinant) < 1e-14:  # Numerische Stabilität prüfen
    raise ValueError("Matrix ist fast singulär")
  solution_0 = (vector[0] * matrix[1, 1] - vector[1] * matrix[0, 1]) / determinant
  solution_1 = (matrix[0, 0] * vector[1] - matrix[1, 0] * vector[0]) / determinant
  return np.array([solution_0, solution_1], dtype=float)


def solve(matrix, vector):
  """
    Löst ein lineares Gleichungssystem. Für kleine Systeme (2x2) wird
    eine geschlossene Lösung verwendet, für größere NumPy-Solver.

    Parameter:
    matrix: np.array, Koeffizientenmatrix
    vector: np.array, rechte Seite des Gleichungssystems

    Rückgabe:
    np.array: Lösung des Gleichungssystems
    """
  return solve_2x2(matrix, vector) if matrix.shape == (2, 2) else np.linalg.solve(matrix, vector)


# ---------- 2. GAUSS-NEWTON (undämpft) ----------------------
def gauss_newton(func, jacobian, x_vals, y_vals, initial_guess, max_iterations=20, tolerance=1e-8):
  """
    Implementiert das ungedämpfte Gauss-Newton-Verfahren zur nichtlinearen
    Optimierung.

    Parameter:
    func: Callable, Modellfunktion
    jacobian: Callable, Jacobi-Matrix der Funktion
    x_vals: np.array, x-Werte der Messdaten
    y_vals: np.array, y-Werte der Messdaten
    initial_guess: np.array, Startschätzung der Parameter
    max_iterations: int, maximale Anzahl von Iterationen
    tolerance: float, Abbruchkriterium basierend auf der Schrittgröße

    Rückgabe:
    tuple: Optimierte Parameter, Verlauf der Iterationen (als Liste von Dictionaries)
    """
  params = np.asarray(initial_guess, dtype=float)
  iteration_history = []
  for iteration in range(max_iterations):
    residuals = y_vals - func(params, x_vals)  # Berechnung der Residuen
    jacobi_matrix = jacobian(params, x_vals)  # Berechnung der Jacobi-Matrix
    normal_matrix = jacobi_matrix.T @ jacobi_matrix  # Normalgleichungsmatrix
    right_hand_side = jacobi_matrix.T @ residuals  # Rechte Seite des Gleichungssystems
    step = solve(normal_matrix, right_hand_side)  # Berechnung der Korrektur
    iteration_history.append({
      "iteration": iteration,
      "params": params.copy(),
      "step": step.copy(),
      "error": np.linalg.norm(residuals)
    })
    if np.linalg.norm(step) < tolerance:  # Überprüfung des Abbruchkriteriums
      return params + step, iteration_history
    params += step
  return params, iteration_history  # Rückgabe nach Maximum an Iterationen


# ---------- 3. GEDÄMPFTES GAUSS-NEWTON ----------------------
def gauss_newton_damped(func, jacobian, x_vals, y_vals, initial_guess, max_iterations=20, tolerance=1e-8,
                        max_halving=8):
  """
    Implementiert das gedämpfte Gauss-Newton-Verfahren. Die Schrittgröße
    wird schrittweise reduziert, falls der Fehler nicht abnimmt.

    Parameter:
    func: Callable, Modellfunktion
    jacobian: Callable, Jacobi-Matrix der Funktion
    x_vals: np.array, x-Werte der Messdaten
    y_vals: np.array, y-Werte der Messdaten
    initial_guess: np.array, Startschätzung der Parameter
    max_iterations: int, maximale Anzahl von Iterationen
    tolerance: float, Abbruchkriterium basierend auf der Schrittgröße
    max_halving: int, maximale Anzahl der Halbierungen der Schrittgröße

    Rückgabe:
    tuple: Optimierte Parameter, Verlauf der Iterationen (als Liste von Dictionaries)
    """
  params = np.asarray(initial_guess, dtype=float)
  iteration_history = []
  for iteration in range(max_iterations):
    residuals = y_vals - func(params, x_vals)
    jacobi_matrix = jacobian(params, x_vals)
    normal_matrix = jacobi_matrix.T @ jacobi_matrix
    right_hand_side = jacobi_matrix.T @ residuals
    step = solve(normal_matrix, right_hand_side)
    old_error = np.linalg.norm(residuals)

    # Dämpfung: Schrittweise Halbierung des Schritts λ = 1, 1/2, 1/4, …
    damping_factor = 1.0
    for _ in range(max_halving + 1):
      trial_params = params + damping_factor * step
      new_error = np.linalg.norm(y_vals - func(trial_params, x_vals))
      if new_error < old_error:
        break  # Schritt akzeptieren
      damping_factor *= 0.5  # Schrittgröße halbieren
    else:
      print("Kein Fortschritt – Abbruch.")
      return params, iteration_history

    iteration_history.append({
      "iteration": iteration,
      "params": params.copy(),
      "step": (damping_factor * step).copy(),
      "old_error": old_error,
      "new_error": new_error,
      "damping_factor": damping_factor
    })
    if np.linalg.norm(damping_factor * step) < tolerance:
      return trial_params, iteration_history
    params = trial_params
  return params, iteration_history


# ---------- 4. MODELLFUNKTION: Exponentielles Wachstum --------
# Messdaten
x_data = np.array([0, 1, 2, 3, 4], dtype=float)
y_data = np.array([3, 1, 0.5, 0.2, 0.05], dtype=float)

# Symbolisches Modell: f(x) = a * exp(b * x)
param_a, param_b, input_x = sp.symbols("a b x")
model_function_symbolic = param_a * sp.exp(param_b * input_x)

# Symbolische Ableitungen für die Jacobi-Matrix
partial_derivative_a = sp.diff(model_function_symbolic, param_a)  # Abgeleitet nach a
partial_derivative_b = sp.diff(model_function_symbolic, param_b)  # Abgeleitet nach b

# Umwandlung zu NumPy-kompatiblen Funktionen
model_function_np = sp.lambdify((param_a, param_b, input_x), model_function_symbolic, "numpy")
partial_a_np = sp.lambdify((param_a, param_b, input_x), partial_derivative_a, "numpy")
partial_b_np = sp.lambdify((param_a, param_b, input_x), partial_derivative_b, "numpy")


def model_function(params, x_values):
  """
    Bewertet die Modellfunktion für gegebene Parameter.
    """
  a, b = params
  return model_function_np(a, b, x_values)


def jacobian_matrix(params, x_values):
  """
    Berechnet die Jacobi-Matrix der Modellfunktion.
    """
  a, b = params
  column1 = partial_a_np(a, b, x_values)  # ∂f/∂a
  column2 = partial_b_np(a, b, x_values)  # ∂f/∂b
  return np.column_stack((column1, column2))

if __name__ == "__main__":
  # Startwerte für die Parameter
  initial_parameters = np.array([1.0, -1.5])

  # ---------- 5. OPTIMIERUNGSAUSFÜHRUNG ------------------------
  # Optimierung mit ungedämpftem Gauss-Newton
  optimized_params_gn, history_gn = gauss_newton(
    model_function, jacobian_matrix, x_data, y_data, initial_parameters
  )

  # Optimierung mit gedämpftem Gauss-Newton
  optimized_params_dgn, history_dgn = gauss_newton_damped(
    model_function, jacobian_matrix, x_data, y_data, initial_parameters
  )

  # Ergebnisse
  print("--------- Ergebnis (ohne Dämpfung) ---------")
  print(f"a = {optimized_params_gn[0]:.6f},  b = {optimized_params_gn[1]:.6f},  Schritte = {len(history_gn)}\n")

  print("--------- Ergebnis (gedämpft) --------------")
  print(f"a = {optimized_params_dgn[0]:.6f},  b = {optimized_params_dgn[1]:.6f},  Schritte = {len(history_dgn)}")