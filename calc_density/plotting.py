import os

import matplotlib.pyplot as plt

from logger import logger


def plot_three_densities(
    x,
    f,
    f_min,
    f_max,
    areas,
    k,
    theta,
    N
):
    plt.figure(figsize=(11, 7))

    plt.plot(
        x,
        f_max,
        linewidth=2,
        label=r"$f_{\max}(x)$ — максимум"
    )

    plt.plot(
        x,
        f_min,
        linewidth=2,
        label=r"$f_{\min}(x)$ — минимум"
    )

    plt.plot(
        x,
        f,
        linewidth=2,
        label=r"$f(x)$ — исходное распределение"
    )

    f_area = areas["f"]
    f_min_area = areas["f_min"]
    f_max_area = areas["f_max"]

    plt.fill_between(
        x[f_area["left"]:f_area["right"] + 1],
        f[f_area["left"]:f_area["right"] + 1],
        alpha=0.25
    )

    plt.fill_between(
        x[f_min_area["left"]:f_min_area["right"] + 1],
        f_min[f_min_area["left"]:f_min_area["right"] + 1],
        alpha=0.25
    )

    plt.fill_between(
        x[f_max_area["left"]:f_max_area["right"] + 1],
        f_max[f_max_area["left"]:f_max_area["right"] + 1],
        alpha=0.25
    )

    plt.title(
        f"Сравнение распределений: "
        f"k={k}, θ={theta}, N={N}",
        fontsize=15
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

    logger.info(
        "График сохранён: %s",
        graph_file
    )

    return graph_file