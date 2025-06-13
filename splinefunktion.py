import numpy as np


def natural_cubic_spline(x, y):
  """
  Berechnet die natürliche kubische Spline-Funktion für gegebene Stützpunkte.

  Parameter:
  x (list): Liste der x-Werte der Stützpunkte (muss aufsteigend sortiert sein).
  y (list): Liste der y-Werte der Stützpunkte.

  Rückgabe:
  list: Liste der Koeffizienten (a, b, c, d) für jedes Intervall [x_i, x_{i+1}].
  """
  n = len(x) - 1  # Anzahl der Intervalle
  h = [x[i + 1] - x[i] for i in range(n)]  # Schrittweiten zwischen den Stützpunkten

  # Initialisiere das tridiagonale System
  A = np.zeros((n + 1, n + 1))  # Koeffizientenmatrix
  b = np.zeros(n + 1)  # Rechte Seite

  # Randbedingungen für die natürliche Spline (zweite Ableitung an den Rändern = 0)
  A[0, 0] = 1  # Zweite Ableitung bei x_0 ist 0
  A[n, n] = 1  # Zweite Ableitung bei x_n ist 0

  # Fülle die tridiagonale Matrix und die rechte Seite
  for i in range(1, n):
    A[i, i - 1] = h[i - 1]  # Unterdiagonale
    A[i, i] = 2 * (h[i - 1] + h[i])  # Hauptdiagonale
    A[i, i + 1] = h[i]  # Oberdiagonale
    # Rechte Seite: Differenzenquotienten der ersten Ableitungen
    b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

  # Löse das lineare Gleichungssystem für die zweiten Ableitungen
  m = np.linalg.solve(A, b)

  # Berechne die Koeffizienten der kubischen Polynome
  splines = []
  for i in range(n):
    a = y[i]  # Konstanter Term
    b = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * m[i] + m[i + 1]) / 3  # Linearer Term
    c = m[i]  # Quadratischer Term
    d = (m[i + 1] - m[i]) / (3 * h[i])  # Kubischer Term
    splines.append((a, b, c, d))  # Speichere die Koeffizienten

  return splines


def evaluate_spline(splines, x_points, x):
  """
  Wertet die Spline-Funktion an einem gegebenen Punkt aus.

  Parameter:
  splines (list): Liste der Koeffizienten (a, b, c, d) für jedes Intervall.
  x_points (list): Liste der x-Werte der Stützpunkte.
  x (float): Der Punkt, an dem die Spline ausgewertet werden soll.

  Rückgabe:
  float: Der interpolierte Wert der Spline-Funktion an der Stelle x.

  Ausnahme:
  ValueError: Wenn x außerhalb des Interpolationsbereichs liegt.
  """
  # Finde das Intervall [x_i, x_{i+1}], in dem x liegt
  for i in range(len(x_points) - 1):
    if x_points[i] <= x <= x_points[i + 1]:
      a, b, c, d = splines[i]
      dx = x - x_points[i]  # Abstand zum linken Intervallrand
      # Berechne den Wert der Spline-Funktion
      return a + b * dx + c * dx ** 2 + d * dx ** 3
  # Fehler, wenn x außerhalb des Bereichs liegt
  raise ValueError("x is out of the interpolation range")


if __name__ == "__main__":
  # Gegebene Stützpunkte (Jahre und Bevölkerungszahlen in Millionen)
  t = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
  p = [75.995, 91.972, 105.711, 123.203, 131.669, 150.697, 179.323, 203.212, 226.506, 249.683, 281.422]

  # Berechne die natürliche kubische Spline
  splines = natural_cubic_spline(t, p)

  # Beispiel: Schätze die Bevölkerungszahl im Jahr 1955
  year = 1955
  estimated_population = evaluate_spline(splines, t, year)
  print(f"The estimated population in {year} is {estimated_population:.3f} million.")

# if __name__ == "__main__":
#     # Example support points
#     x = [0, 1, 2, 3]
#     y = [1, 2, 0, 2]
#
#     # Compute the natural cubic spline
#     splines = natural_cubic_spline(x, y)
#
#     # Print the cubic polynomials
#     for i, (a, b, c, d) in enumerate(splines):
#         print(f"S_{i}(x) = {a:.2f} + {b:.2f}(x - {x[i]}) + {c:.2f}(x - {x[i]})^2 + {d:.2f}(x - {x[i]})^3")
