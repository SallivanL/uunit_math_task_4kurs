import os
import matplotlib.pyplot as plt

from calc import calculate_densities, find_peak_area
from logger import log
from points import save_plot_points


def plot_three_densities(k, theta, N):

    x, f, phi_max, psi_min = calculate_densities(
        k=k,
        theta=theta,
        N=N
    )

    m = 0.7

    f_area = find_peak_area(x, f, m)
    phi_area = find_peak_area(x, phi_max, m)
    psi_area = find_peak_area(x, psi_min, m)

    # Сохраняем точки в общий log.md
    points_file = save_plot_points(
        x=x,
        f=f,
        phi_max=phi_max,
        psi_min=psi_min,
        k=k,
        theta=theta,
        N=N,
        save_points=100
    )

    # Построение графика
    plt.figure(figsize=(11, 7))

    plt.plot(
        x,
        phi_max,
        linewidth=2,
        label=r"$\varphi_{\max}(x)$ — максимум"
    )

    plt.plot(
        x,
        psi_min,
        linewidth=2,
        label=r"$\psi_{\min}(x)$ — минимум"
    )

    plt.plot(
        x,
        f,
        linewidth=2,
        label=r"$f(x)$ — исходное распределение"
    )

    plt.title(
        f"Сравнение распределений: k={k}, θ={theta}, N={N}",
        fontsize=15
    )

    plt.fill_between(
        x[f_area["left"]:f_area["right"] + 1],
        f[f_area["left"]:f_area["right"] + 1],
        alpha=0.25
    )

    plt.fill_between(
        x[phi_area["left"]:phi_area["right"] + 1],
        phi_max[phi_area["left"]:phi_area["right"] + 1],
        alpha=0.25
    )

    plt.fill_between(
        x[psi_area["left"]:psi_area["right"] + 1],
        psi_min[psi_area["left"]:psi_area["right"] + 1],
        alpha=0.25
    )
    plt.xlabel("x", fontsize=13)
    plt.ylabel("Плотность", fontsize=13)

    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=11)

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)

    graph_file = (
        f"results/"
        f"distribution_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.png"
    )

    plt.savefig(
        graph_file,
        dpi=150
    )

    plt.close()

    log(
        "График сохранён",
        graph=graph_file,
        points=points_file,
        k=k,
        theta=theta,
        N=N
    )