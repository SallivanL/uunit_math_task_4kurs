from scipy.stats import gamma

from logger import log


def calculate_densities(k, theta, N, points=5000):

    if k <= 0:
        raise ValueError("k должно быть > 0")

    if theta <= 0:
        raise ValueError("theta должно быть > 0")

    if N <= 0:
        raise ValueError("N должно быть > 0")

    N = int(N)

    log(
        "Начало расчёта",
        k=k,
        theta=theta,
        N=N
    )

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

    # Исходная плотность
    f = gamma.pdf(
        x,
        a=k,
        scale=theta
    )

    # CDF
    F = gamma.cdf(
        x,
        a=k,
        scale=theta
    )

    # Плотность максимума
    phi_max = (
        N
        * f
        * np.power(F, N - 1)
    )

    # Плотность минимума
    psi_min = (
        N
        * f
        * np.power(1 - F, N - 1)
    )

    log(
        "Расчёт завершён",
        k=k,
        theta=theta,
        N=N,
        f_max=np.max(f),
        phi_max=np.max(phi_max),
        psi_min=np.max(psi_min)
    )

    return x, f, phi_max, psi_min


import numpy as np
from scipy.integrate import cumulative_trapezoid


def find_peak_area(x, y, m=0.7):
    peak_index = np.argmax(y)

    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    left = peak_index
    right = peak_index

    while left > 0 or right < len(x) - 1:
        left_area = np.trapezoid(y[left:peak_index + 1],
                                 x[left:peak_index + 1])
        right_area = np.trapezoid(y[peak_index:right + 1],
                                  x[peak_index:right + 1])

        current_area = left_area + right_area

        if current_area >= target_area:
            break

        left_candidate = -1
        right_candidate = -1

        if left > 0:
            left_candidate = y[left - 1]

        if right < len(x) - 1:
            right_candidate = y[right + 1]

        if left_candidate >= right_candidate and left > 0:
            left -= 1
        elif right < len(x) - 1:
            right += 1
        else:
            break

    area = np.trapezoid(y[left:right + 1], x[left:right + 1])

    return {
        "left": left,
        "right": right,
        "peak": peak_index,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area
    }