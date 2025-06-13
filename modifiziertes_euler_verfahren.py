import matplotlib.pyplot as plt


def modified_euler_method(f, t0, y0, t_end, n):
  """
  Implementiert das modifizierte Euler-Verfahren zur Lösung von y'(t) = f(t, y).

  Parameter:
  f : Funktion, die die rechte Seite der Differentialgleichung definiert f(t, y)
  t0 : Startzeit t0
  y0 : Anfangswert y(t0)
  t_end : Endzeit t = t_end
  n : Anzahl der Schritte im Intervall [t0, t_end]

  Rückgabe:
  t_values : Liste von Zeitpunkten
  y_values : Liste der Lösung y(t) an diesen Zeitpunkten
  """
  h = (t_end - t0) / n  # Schrittweite h
  t_values = [t0 + i * h for i in range(n + 1)]  # Zeitpunkte für n + 1 Werte
  y_values = [0] * (n + 1)  # Initialisiert mit Nullen
  y_values[0] = y0  # Startwert y(t0)

  # Modifizierter Euler-Schritt für alle Zeitwerte
  for i in range(n):
    y_predict = y_values[i] + h * f(t_values[i], y_values[i])  # Vorhersage
    y_values[i + 1] = y_values[i] + (h / 2) * (f(t_values[i], y_values[i]) + f(t_values[i + 1], y_predict))  # Korrektur
  return t_values, y_values


# Rechte Seite der Differentialgleichung y'(t) = t^2 + 0.1y
def f(t, y):
  return t ** 2 + 0.1 * y


# Exakte Lösung der DGL: y(t) = -10t^2 - 200t - 2000 + 1722.5 * e^(0.05 * (2t + 3))
def exact_solution(t):
  from math import exp
  return -10 * t ** 2 - 200 * t - 2000 + 1722.5 * exp(0.05 * (2 * t + 3))


if __name__ == "__main__":
  # Parameter der Aufgabe
  t0 = -1.5  # Startwert t = -1.5
  y0 = 0  # Anfangswert y(t0) = 0
  t_end = 1.5  # Endzeit t = 1.5
  n = 5  # Anzahl der Intervalle (Schritte)

  # Modifiziertes Euler-Verfahren ausführen
  t_values, y_values = modified_euler_method(f, t0, y0, t_end, n)

  # Exakte Lösung berechnen (Referenzwerte)
  y_exact = [exact_solution(t) for t in t_values]

  # Visualisierung der Lösungen
  plt.figure(figsize=(10, 6))
  plt.plot(t_values, y_values, 'r-o', label="Modifiziertes Euler-Approximation")
  plt.plot(t_values, y_exact, 'b-', label="Exakte Lösung")
  plt.title("Modifiziertes Euler-Verfahren und exakte Lösung für $y'(t) = t^2 + 0.1y$")
  plt.xlabel("$t$")
  plt.ylabel("$y$")
  plt.grid(True)
  plt.legend()
  plt.show()

  # Fehler berechnen (absolute Differenz)
  errors = [abs(y_e - y_a) for y_e, y_a in zip(y_exact, y_values)]

  # Visualisierung der Fehleranalyse
  plt.figure(figsize=(10, 6))
  plt.plot(t_values, errors, 'g-o', label="Absolute Fehler")
  plt.title("Fehleranalyse des Modifizierten Euler-Verfahrens")
  plt.xlabel("$t$")
  plt.ylabel("Fehler")
  plt.grid(True)
  plt.legend()
  plt.show()
