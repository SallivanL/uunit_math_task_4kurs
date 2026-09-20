import numpy as np
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