import numpy as np
from scipy.stats import gamma


def calculate_densities(k, theta, N, points=5000):
    if k <= 0:
        raise ValueError("k должно быть > 0")

    if theta <= 0:
        raise ValueError("theta должно быть > 0")

    if N <= 0:
        raise ValueError("N должно быть > 0")

    N = int(N)

    x_max = gamma.ppf(
        0.9999,
        a=k,
        scale=theta
    )

    x = np.linspace(
        0,
        x_max,
        points
    )

    f = gamma.pdf(
        x,
        a=k,
        scale=theta
    )

    F = gamma.cdf(
        x,
        a=k,
        scale=theta
    )

    f_max = (
        N
        * f
        * np.power(F, N - 1)
    )

    f_min = (
        N
        * f
        * np.power(1 - F, N - 1)
    )

    return x, f, f_min, f_max


def calculate_area(x, y, m):
    peak = np.argmax(y)

    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    left = peak
    right = peak

    while left > 0 or right < len(x) - 1:
        left_area = np.trapezoid(
            y[left:peak + 1],
            x[left:peak + 1]
        )

        right_area = np.trapezoid(
            y[peak:right + 1],
            x[peak:right + 1]
        )

        current_area = left_area + right_area

        if current_area >= target_area:
            break

        left_candidate = y[left - 1] if left > 0 else -1
        right_candidate = y[right + 1] if right < len(x) - 1 else -1

        if left_candidate >= right_candidate and left > 0:
            left -= 1
        elif right < len(x) - 1:
            right += 1
        else:
            break

    area = np.trapezoid(
        y[left:right + 1],
        x[left:right + 1]
    )

    return {
        "left": left,
        "right": right,
        "peak": peak,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area
    }


def calculate_areas(x, f, f_min, f_max, m):
    return {
        "f": calculate_area(x, f, m),
        "f_min": calculate_area(x, f_min, m),
        "f_max": calculate_area(x, f_max, m)
    }