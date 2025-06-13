import numpy as np
import matplotlib.pyplot as plt


def euler_method(f, t0, y0, t_end, n):
  """
    Implementiert das klassische Euler-Verfahren zur Lösung von y'(t) = f(t, y).

    Parameter:
    f : Funktion, die die rechte Seite der Differentialgleichung definiert f(t, y)
    t0 : Startzeit t0
    y0 : Anfangswert y(t0)
    t_end : Endzeit t = t_end
    n : Anzahl der Schritte im Intervall [t0, t_end]

    Rückgabe:
    t_values : Array von Zeitpunkten
    y_values : Array der Lösung y(t) an diesen Zeitpunkten
    """
  h = (t_end - t0) / n  # Schrittweite h
  t_values = np.linspace(t0, t_end, n + 1)  # Zeitpunkte für n + 1 Werte
  y_values = np.zeros(n + 1)  # Initialisiert mit Nullen
  y_values[0] = y0  # Startwert y(t0)

  # Euler-Schritt für alle Zeitwerte
  for i in range(n):
    y_values[i + 1] = y_values[i] + h * f(t_values[i], y_values[i])  # Formel des Euler-Verfahrens
  return t_values, y_values


# Rechte Seite der Differentialgleichung y'(t) = t^2 + 0.1y
def f(t, y):
  return t ** 2 + 0.1 * y


# Exakte Lösung der DGL: y(t) = -10t^2 - 200t - 2000 + 1722.5 * e^(0.05 * (2t + 3))
def exact_solution(t):
  return -10 * t ** 2 - 200 * t - 2000 + 1722.5 * np.exp(0.05 * (2 * t + 3))


def bit_error_analysis(errors, bits=16):
  """
    Überprüft, ob der Fehler unter der gegebenen Bit-Genauigkeit liegt.

    Parameter:
    errors : Array mit Fehlerwerten
    bits : Anzahl der signifikanter Bits

    Rückgabe:
    bit_error_flags : Array mit True/False, ob Fehler unter der verlangten Genauigkeit liegt
    """
  # Maximale Genauigkeit basierend auf der Bitanzahl (Maschinengenauigkeitsschwelle)
  tolerance = 2 ** (-bits)
  return errors < tolerance


if __name__ == "__main__":
  # Parameter der Aufgabe
  t0 = -1.5  # Startwert t = -1.5
  y0 = 0  # Anfangswert y(t0) = 0
  t_end = 1.5  # Endzeit t = 1.5
  n = 5  # Anzahl der Intervalle (Schritte)
  bit_precision = 16  # Eingestellte Anzahl der Bits (Genauigkeit)

  # Euler-Verfahren ausführen
  t_values, y_values = euler_method(f, t0, y0, t_end, n)

  # Exakte Lösung berechnen (Referenzwerte)
  y_exact = exact_solution(t_values)

  # Fehler berechnen (absolute Differenz)
  errors = np.abs(y_exact - y_values)

  # Bitabhängige Fehleranalyse
  bit_errors = bit_error_analysis(errors, bits=bit_precision)

  # Ergebnisse in der Konsole ausgeben
  print("t-Werte:", t_values)
  print("y-Werte (Euler, numerisch):", y_values)
  print("y-Werte (exakte Lösung):", y_exact)
  print("Fehler:", errors)
  print(f"Fehler < 2^(-{bit_precision}):", bit_errors)

  # Visualisierung der Lösungen
  plt.figure(figsize=(10, 6))
  plt.plot(t_values, y_values, 'r-o', label="Euler-Approximation")
  plt.plot(t_values, y_exact, 'b-', label="Exakte Lösung")
  plt.title("Euler-Verfahren und exakte Lösung für $y'(t) = t^2 + 0.1y$")
  plt.xlabel("$t$")
  plt.ylabel("$y$")
  plt.grid(True)
  plt.legend()
  plt.show()

  # Visualisierung der Fehleranalyse
  plt.figure(figsize=(10, 6))
  plt.plot(t_values, errors, 'g-o', label="Absolute Fehler")
  plt.axhline(2 ** (-bit_precision), color='r', linestyle='--',
              label=f"Fehlertoleranz (2^(-{bit_precision}))")
  plt.title(f"Fehleranalyse mit {bit_precision} Bits Präzision")
  plt.xlabel("$t$")
  plt.ylabel("Fehler")
  plt.grid(True)
  plt.legend()
  plt.show()