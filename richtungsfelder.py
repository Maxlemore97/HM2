import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def draw_direction_field_and_solutions(f, x_vals, y_vals, initial_conditions_list):
  """
  Zeichnet das Richtungsfeld für die Differentialgleichung y'(x) = f(x, y),
  sowie beliebig viele Lösungskurven für gegebene Anfangswerte.

  Parameter:
  f : Funktion, die die Differentialgleichung definiert (y'(x) = f(x, y))
  x_vals : Liste von x-Werten (Rasterpunkte)
  y_vals : Liste von y-Werten (Rasterpunkte)
  initial_conditions_list : Liste der Anfangsbedingungen [(x0, y0), ...]
  """
  # Erstellen des Gitters
  X, Y = np.meshgrid(x_vals, y_vals)

  # Steigungen berechnen
  U = np.ones_like(X)  # Einheitlich 1 für Pfeile in x-Richtung
  V = f(X, Y)  # Steigungen in y-Richtung

  # Pfeile normalisieren, um gleichmäßig zu skalieren
  magnitude = np.sqrt(U ** 2 + V ** 2)
  U /= magnitude
  V /= magnitude

  # Zeichnen des Richtungsfeldes
  plt.figure(figsize=(10, 8))
  plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='b', alpha=0.5, label="Richtungsfeld")

  # Zeichnen der Lösungskurven für alle Anfangswerte
  for x0, y0 in initial_conditions_list:
    # Intervall für numerische Integration erstellen
    t_left = np.linspace(x0, x_vals[0], 500)  # Bereich nach links (Rückwärtsintegration)
    t_right = np.linspace(x0, x_vals[-1], 500)  # Bereich nach rechts (Vorwärtsintegration)

    # Definition der DGL für scipy.integrate.odeint
    def dgl(y, t):
      return f(t, y)

    # Lösung der DGL nach links und rechts
    solution_left = odeint(dgl, y0, t_left)
    solution_right = odeint(dgl, y0, t_right)

    # Zeichnen der Lösungskurven (rot) – links und rechts
    plt.plot(t_left, solution_left, 'r-', linewidth=2, alpha=0.8)
    plt.plot(t_right, solution_right, 'r-', linewidth=2, alpha=0.8, label=f"Lösung: y({x0}) = {y0}")

  # Plot-Labels und -Details
  plt.title(r"Richtungsfeld und Lösungen: $y'(x) = x^2 + 0.1 \cdot y(x)$")
  plt.xlabel("$x$")
  plt.ylabel("$y$")
  plt.axhline(0, color='black', linewidth=0.5)  # x-Achse
  plt.axvline(0, color='black', linewidth=0.5)  # y-Achse
  plt.grid(True)
  plt.xlim([x_vals[0], x_vals[-1]])
  plt.ylim([y_vals[0], y_vals[-1]])
  plt.legend()
  plt.show()


if __name__ == "__main__":
  # Differentialgleichung definieren
  def f(x, y):
    return x ** 2 + 0.1 * y  # Die korrekte Formel des Problems


  # Bereich der x- und y-Werte (Rasterpunkte für das Richtungsfeld)
  x_vals = np.linspace(-3, 3, 25)  # Bereich der x-Werte
  y_vals = np.linspace(-2, 4, 25)  # Bereich der y-Werte

  # Anfangsbedingungen: Liste von Anfangspunkten
  initial_conditions = [
    (-1.5, 0),  # Anfangswert y(-1.5) = 0
    (0, 0.5)  # Anfangswert y(0) = 0.5
  ]

  # Aufruf der Funktion zum Zeichnen des Richtungsfelds und der Lösungskurven
  draw_direction_field_and_solutions(f, x_vals, y_vals, initial_conditions)
