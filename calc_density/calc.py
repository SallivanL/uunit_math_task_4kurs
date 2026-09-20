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

def find_peak_area(x, y, m):
    peak_index = np.argmax(y)

    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    left = peak_index
    right = peak_index

    while left > 0 or right < len(x) - 1:
        left_area = np.trapezoid(
            y[left:peak_index + 1],
            x[left:peak_index + 1]
        )

        right_area = np.trapezoid(
            y[peak_index:right + 1],
            x[peak_index:right + 1]
        )

        if left_area + right_area >= target_area:
            break

        left_candidate = (
            y[left - 1]
            if left > 0
            else -1
        )

        right_candidate = (
            y[right + 1]
            if right < len(x) - 1
            else -1
        )

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
        "peak": peak_index,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area
    }

def find_left_area(x, y, m):
    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    right = 0

    for i in range(1, len(x)):
        area = np.trapezoid(
            y[:i + 1],
            x[:i + 1]
        )

        if area >= target_area:
            right = i
            break

    area = np.trapezoid(
        y[:right + 1],
        x[:right + 1]
    )

    return {
        "left": 0,
        "right": right,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area
    }

def find_right_area(x, y, m):
    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    left = len(x) - 1

    for i in range(len(x) - 2, -1, -1):
        area = np.trapezoid(
            y[i:],
            x[i:]
        )

        if area >= target_area:
            left = i
            break

    area = np.trapezoid(
        y[left:],
        x[left:]
    )

    return {
        "left": left,
        "right": len(x) - 1,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area
    }


def calculate_areas(x, f, f_min, f_max, m):
    return {
        "f": find_peak_area(x, f, m),
        "f_min": find_left_area(x, f_min, m),
        "f_max": find_right_area(x, f_max, m),
    }