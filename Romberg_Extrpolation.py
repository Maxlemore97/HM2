import numpy as np


def f(x):
  """Die gegebene Integrationsfunktion f(x) = 1 / x."""
  return 1 / x


def romberg_extrapolation(f, a, b, levels):
  """
    Berechnet das Integral von f(x) im Intervall [a, b] mit der Romberg-Extrapolation.

    Parameter:
    f : Funktion, die zu integrieren ist.
    a : float, untere Grenze des Integrals.
    b : float, obere Grenze des Integrals.
    levels : int, Anzahl der Stufen (z.B. 4 für eine 4x4 Romberg-Tabelle).

    Rückgabe:
    tuple: Die vollständige Romberg-Tabelle und das finale Ergebnis.
    """
  # Romberg-Tabelle initialisieren (Quadratische Matrix mit Nullwerten)
  romberg_table = np.zeros((levels, levels))

  # Berechnung der Trapezregel-Resultate für jede Stufe
  for j in range(levels):
    n = 2 ** j  # Anzahl der Teilintervalle (2^j)
    h = (b - a) / n  # Schrittweite
    # Trapezregel: Summe der Funktionswerte an Zwischenpunkten
    sum_midpoints = sum(f(a + i * h) for i in range(1, n))  # Ausschließlich Zwischenpunkte
    romberg_table[j, 0] = h * (0.5 * (f(a) + f(b)) + sum_midpoints)  # Trapezregel für j-te Stufe

  # Romberg-Extrapolation für jede weitere Spalte...
  for k in range(1, levels):
    for j in range(k, levels):
      romberg_table[j, k] = ((4 ** k) * romberg_table[j, k - 1] - romberg_table[j - 1, k - 1]) / (4 ** k - 1)

  return romberg_table


if __name__ == "__main__":
  # Integrationsgrenzen
  a, b = 2, 4

  # Exakter Wert des Integrals: ln(2)
  exact_integral = np.log(2)  # ≈ 0.6931471806

  # Anzahl der Stufen für die Romberg-Extrapolation
  levels = 5

  # Erstellung der Romberg-Tabelle
  romberg_table = romberg_extrapolation(f, a, b, levels)

  # Tabelle ausgeben
  print("Romberg-Extrapolationstabelle:")
  for i in range(levels):
    row = " | ".join(f"{romberg_table[i, k]:.10f}" for k in range(i + 1))  # Nur berechnete Spalten ausgeben
    print(row)

  # Bestes Ergebnis aus der Romberg-Tabelle (unten rechts)
  final_result = romberg_table[levels - 1, levels - 1]

  # Ergebnisse
  print(f"\nFinales Ergebnis der Romberg-Extrapolation (Stufe {levels}): {final_result:.10f}")
  print(f"Exakter Wert des Integrals: {exact_integral:.10f}")
  print(f"Absoluter Fehler: {abs(final_result - exact_integral):.10e}")