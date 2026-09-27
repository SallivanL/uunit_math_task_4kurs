# plotting.py

import matplotlib.pyplot as plt


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
    figure = plt.figure(
        figsize=(11, 7)
    )

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

    # Оранжевая область
    plt.fill_between(
        x[
            f_min_area["left"]:
            f_min_area["right"] + 1
        ],
        f_min[
            f_min_area["left"]:
            f_min_area["right"] + 1
        ],
        alpha=0.25
    )

    # Синяя область
    plt.fill_between(
        x[
            f_area["left"]:
            f_area["right"] + 1
        ],
        f[
            f_area["left"]:
            f_area["right"] + 1
        ],
        alpha=0.25
    )

    # Зелёная область
    plt.fill_between(
        x[
            f_max_area["left"]:
            f_max_area["right"] + 1
        ],
        f_max[
            f_max_area["left"]:
            f_max_area["right"] + 1
        ],
        alpha=0.25
    )

    plt.title(
        f"Сравнение распределений: "
        f"k={k}, θ={theta}, N={N}",
        fontsize=15
    )

    plt.xlabel(
        "x",
        fontsize=13
    )

    plt.ylabel(
        "Плотность",
        fontsize=13
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.legend(
        fontsize=11
    )

    plt.tight_layout()

    return figure