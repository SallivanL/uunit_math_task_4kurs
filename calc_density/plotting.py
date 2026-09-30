import matplotlib.pyplot as plt


def plot_three_densities(
    x,
    f,
    f_min,
    f_max,
    areas,
    k,
    theta,
    N,
):
    """
    Строит три плотности:

        f_max(x) — максимум
        f_min(x) — минимум
        f(x)     — исходное распределение

    Области по x:

        ОРАНЖЕВАЯ | СИНЯЯ | ЗЕЛЁНАЯ

    Границы:

        оранжевая — определяется через f_min и m
        зелёная   — определяется через f_max и m
        синяя     — промежуток между ними
    """

    figure = plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        x,
        f_max,
        linewidth=2,
        label=r"$f_{\max}(x)$ — максимум",
    )

    plt.plot(
        x,
        f_min,
        linewidth=2,
        label=r"$f_{\min}(x)$ — минимум",
    )

    plt.plot(
        x,
        f,
        linewidth=2,
        label=r"$f(x)$ — исходное распределение",
    )

    orange_area = areas["orange"]
    blue_area = areas["blue"]
    green_area = areas["green"]

    # --------------------------------------------------
    # Оранжевая область — f_min
    # --------------------------------------------------

    orange_left = orange_area["left"]
    orange_right = orange_area["right"]

    plt.fill_between(
        x[
            orange_left:
            orange_right + 1
        ],
        f_min[
            orange_left:
            orange_right + 1
        ],
        alpha=0.25,
    )

    # --------------------------------------------------
    # Синяя область — исходное f
    # --------------------------------------------------

    blue_left = blue_area["left"]
    blue_right = blue_area["right"]

    plt.fill_between(
        x[
            blue_left:
            blue_right + 1
        ],
        f[
            blue_left:
            blue_right + 1
        ],
        alpha=0.25,
    )

    # --------------------------------------------------
    # Зелёная область — f_max
    # --------------------------------------------------

    green_left = green_area["left"]
    green_right = green_area["right"]

    plt.fill_between(
        x[
            green_left:
            green_right + 1
        ],
        f_max[
            green_left:
            green_right + 1
        ],
        alpha=0.25,
    )

    plt.title(
        f"Сравнение распределений: "
        f"k={k}, θ={theta}, N={N}",
        fontsize=15,
    )

    plt.xlabel(
        "x",
        fontsize=13,
    )

    plt.ylabel(
        "Плотность",
        fontsize=13,
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend(
        fontsize=11,
    )

    plt.tight_layout()

    return figure


def plot_replay_histogram(
    average_replay_ratios,
    k,
    theta,
    N,
    replays,
):
    """
    Строит одну итоговую гистограмму.

    Для каждой из трёх областей рассчитывается
    средняя доля точек n / N по всем replay.

    Получается ровно три столбца:

        Область 1 → mean(n1 / N)
        Область 2 → mean(n2 / N)
        Область 3 → mean(n3 / N)

    Сумма высот трёх столбцов равна 1.
    """

    figure = plt.figure(
        figsize=(9, 7)
    )

    values = [
        average_replay_ratios["orange_ratio"],
        average_replay_ratios["blue_ratio"],
        average_replay_ratios["green_ratio"],
    ]

    labels = [
        "Область 1",
        "Область 2",
        "Область 3",
    ]

    bars = plt.bar(
        labels,
        values,
        width=0.6,
    )

    # Подписываем значение над каждым столбцом.

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.ylim(
        0,
        1.05,
    )

    plt.ylabel(
        "Средняя доля точек n / N",
        fontsize=13,
    )

    plt.xlabel(
        "Область",
        fontsize=13,
    )

    plt.title(
        "Среднее распределение точек по областям: "
        f"k={k}, θ={theta}, N={N}, "
        f"replays={replays}",
        fontsize=15,
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    return figure