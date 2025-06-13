from math import log, ceil


def rechteckregel(f, a, b, n, include_error=False, max_f2=None):
  """
    Numerische Integration mit der Rechteckregel (unter Verwendung der Mittelpunktsregel).
    Berechnet optional die Fehlergrenze gemäß dem angegebenen Fehlerterm.

    Parameter:
    f : Funktion, die zu integrieren ist.
    a : float, untere Grenze des Integrals.
    b : float, obere Grenze des Integrals.
    n : int, Anzahl der Intervalle.
    include_error : bool, wenn True, wird der Fehler und die Fehlergrenze berechnet.
    max_f2 : float, Maximum von |f''(x)| im Intervall. Notwendig, wenn include_error=True.

    Rückgabe:
    float oder tuple: Die Näherung des Integrals und optional der Fehler.
    """
  h = (b - a) / n
  integral = 0.0

  for i in range(n):
    midpoint = a + h * (i + 0.5)
    integral += f(midpoint)
  integral = h * integral

  if include_error:
    if max_f2 is None:
      raise ValueError("Für die Fehler-Formel wird max_f2 (Maximum von |f''(x)|) benötigt.")
    error_boundary = (h ** 2) / 24 * (b - a) * max_f2
    return integral, error_boundary

  return integral


def trapezregel(f, a, b, n, include_error=False, max_f2=None):
  """
    Numerische Integration mit der Trapezregel.
    Berechnet optional die Fehlergrenze gemäß dem angegebenen Fehlerterm.

    Parameter:
    f : Funktion, die zu integrieren ist.
    a : float, untere Grenze des Integrals.
    b : float, obere Grenze des Integrals.
    n : int, Anzahl der Intervalle.
    include_error : bool, wenn True, wird die Fehlergrenze berechnet.
    max_f2 : float, Maximum von |f''(x)| im Intervall. Notwendig, wenn include_error=True.

    Rückgabe:
    float oder tuple: Die Näherung des Integrals und optional die Fehlergrenze.
    """
  h = (b - a) / n
  integral = 0.5 * (f(a) + f(b))

  for i in range(1, n):
    x_i = a + i * h
    integral += f(x_i)
  integral = h * integral

  if include_error:
    if max_f2 is None:
      raise ValueError("Für die Fehler-Formel wird max_f2 (Maximum von |f''(x)|) benötigt.")
    error_boundary = (h ** 2) / 12 * (b - a) * max_f2
    return integral, error_boundary

  return integral


def simpsonregel(f, a, b, n, include_error=False, max_f4=None):
  """
    Numerische Integration mit der Simpsonregel.
    Berechnet optional die Fehlergrenze gemäß dem angegebenen Fehlerterm.

    Parameter:
    f : Funktion, die zu integrieren ist.
    a : float, untere Grenze des Integrals.
    b : float, obere Grenze des Integrals.
    n : int, Anzahl der Intervalle (muss gerade sein).
    include_error : bool, wenn True, wird die Fehlergrenze berechnet.
    max_f4 : float, Maximum von |f''''(x)| im Intervall. Notwendig, wenn include_error=True.

    Rückgabe:
    float oder tuple: Die Näherung des Integrals und optional die Fehlergrenze.
    """
  if n % 2 == 1:
    raise ValueError("Die Anzahl der Intervalle n muss für die Simpsonregel gerade sein.")

  h = (b - a) / n
  integral = f(a) + f(b)

  for i in range(1, n):
    x_i = a + i * h
    if i % 2 == 0:
      integral += 2 * f(x_i)
    else:
      integral += 4 * f(x_i)
  integral = (h / 3) * integral

  if include_error:
    if max_f4 is None:
      raise ValueError("Für die Fehler-Formel wird max_f4 (Maximum von |f''''(x)|) benötigt.")
    error_boundary = (h ** 4) / 2880 * (b - a) * max_f4
    return integral, error_boundary

  return integral


def gauss_quadratur(f, a, b, n):
  """
    Numerische Integration mit der Gauß-Quadratur für n = 1, 2 oder 3.
    Berechnet das Integral durch eine Gewichtung der Funktionswerte an speziellen Punkten.

    Parameter:
    f : Funktion, die zu integrieren ist.
    a : float, untere Grenze des Integrals.
    b : float, obere Grenze des Integrals.
    n : int, Anzahl der Punkte (n = 1, 2 oder 3).

    Rückgabe:
    float: Die Näherung des Integrals.
    """
  # Nullstellen der Legendre-Polynome und zugehörige Gewichte für [-1, 1]
  if n == 1:
    # Ein Punkt: Gewicht = 2 und Punkt = 0 (Mittelpunktregel)
    points = [0]
    weights = [2]
  elif n == 2:
    # Zwei Punkte: Nullstellen von P2(x) = (1/2)*(3x^2 - 1)
    points = [-1 / (3 ** 0.5), 1 / (3 ** 0.5)]
    weights = [1, 1]
  elif n == 3:
    # Drei Punkte: Nullstellen von P3(x) = (1/2)*(5x^3 - 3x)
    points = [-0.774596669241483, 0, 0.774596669241483]
    weights = [5 / 9, 8 / 9, 5 / 9]
  else:
    raise ValueError("Unterstützt nur für n = 1, 2 oder 3.")

  # Transformation von [-1, 1] auf [a, b]
  def transform(x):
    return ((b - a) / 2) * x + (b + a) / 2

  # Gauß-Quadratur: gewichtete Summe der Funktionswerte an den transformierten Punkten
  integral = 0
  for i in range(n):
    integral += weights[i] * f(transform(points[i]))

  # Den Skalierungsfaktor (b - a) / 2 berücksichtigen
  integral *= (b - a) / 2

  return integral


# Beispiel zur Nutzung der Methoden:
if __name__ == "__main__":
  # Gegebene Funktion: f(x) = 1/x
  def f(x):
    return 1 / x

  # Ableitungsmaxima für Fehlerberechnungen
  max_f2 = 0.25  # Maximum von |f''(x)| im Intervall [2, 4]
  max_f4 = 0.75  # Maximum von |f''''(x)| im Intervall [2, 4]

  # Integrationsbereich und Unterteilungen
  a, b = 2, 4
  n = 4

  # Exakter Wert des Integrals
  exact = log(2)  # ln(2)

  # Rechteckregel
  rechteck_result, rechteck_error_bound = rechteckregel(f, a, b, n, include_error=True, max_f2=max_f2)
  rechteck_actual_error = abs(exact - rechteck_result)
  print(f"Rechteckregel (Mittelpunkt): {rechteck_result:.5f}, Fehlergrenze: {rechteck_error_bound:.5e}, Tatsächlicher Fehler: {rechteck_actual_error:.5e}")

  # Trapezregel
  trapez_result, trapez_error_bound = trapezregel(f, a, b, n, include_error=True, max_f2=max_f2)
  trapez_actual_error = abs(exact - trapez_result)
  print(f"Trapezregel: {trapez_result:.5f}, Fehlergrenze: {trapez_error_bound:.5e}, Tatsächlicher Fehler: {trapez_actual_error:.5e}")

  # Simpsonregel
  simpson_result, simpson_error_bound = simpsonregel(f, a, b, n, include_error=True, max_f4=max_f4)
  simpson_actual_error = abs(exact - simpson_result)
  print(f"Simpsonregel: {simpson_result:.5f}, Fehlergrenze: {simpson_error_bound:.5e}, Tatsächlicher Fehler: {simpson_actual_error:.5e}")

  # Gauß-Quadratur mit 1, 2 und 3 Punkten
  gauss_1_result = gauss_quadratur(f, a, b, n=1)
  gauss_2_result = gauss_quadratur(f, a, b, n=2)
  gauss_3_result = gauss_quadratur(f, a, b, n=3)

  print(f"Gauß-Quadratur (n=1): {gauss_1_result:.6f}")
  print(f"Gauß-Quadratur (n=2): {gauss_2_result:.6f}")
  print(f"Gauß-Quadratur (n=3): {gauss_3_result:.6f}")
