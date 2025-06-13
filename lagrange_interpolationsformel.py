def lagrange_interpolation(t_points, y_points, t):
    n = len(t_points)
    result = 0.0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (t - t_points[j]) / (t_points[i] - t_points[j])
        result += term
    return result

if __name__ == "__main__":
    # Given data points
    t_points = [8, 10, 12, 14]  # Time in hours
    y_points = [11.2, 13.4, 15.3, 19.5]  # Temperature in °C

    # Time at which we want to estimate the temperature
    t = 11

    # Estimate temperature using Lagrange interpolation
    estimated_temp = lagrange_interpolation(t_points, y_points, t)
    print(f"The estimated temperature at {t} o'clock is {estimated_temp:.2f} °C.")