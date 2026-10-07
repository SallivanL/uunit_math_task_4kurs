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


def plot_cdf(
    x,
    cdf_area_probabilities,
    areas,
    k,
    theta,
    N,
):
    """
    Строит функцию распределения F(x) исходного
    Gamma-распределения и отмечает на ней те же
    две границы, которые разделяют три области.

    Для каждой границы проводится вертикальная
    проекция до оси X и горизонтальная проекция
    до значения F(x).

    Точки:

        x1 = граница оранжевой области
        x2 = граница зелёной области

        F(x1)
        F(x2)
    """

    figure = plt.figure(
        figsize=(11, 7)
    )

    x1 = cdf_area_probabilities[
        "orange_boundary"
    ]

    x2 = cdf_area_probabilities[
        "green_boundary"
    ]

    cdf_values = cdf_area_probabilities.get(
        "cdf_values"
    )

    if cdf_values is None:
        raise ValueError(
            "cdf_values отсутствует в результатах "
            "расчёта CDF."
        )

    cdf_x1 = cdf_area_probabilities[
        "cdf_orange"
    ]

    cdf_x2 = cdf_area_probabilities[
        "cdf_green"
    ]

    plt.plot(
        x,
        cdf_values,
        linewidth=2,
        label=r"$F(x)$ — функция распределения",
    )

    plt.scatter(
        [x1, x2],
        [cdf_x1, cdf_x2],
        s=55,
        zorder=5,
        label="Точки границ областей",
    )

    plt.vlines(
        x1,
        0,
        cdf_x1,
        linestyles="--",
        linewidth=1.5,
    )

    plt.vlines(
        x2,
        0,
        cdf_x2,
        linestyles="--",
        linewidth=1.5,
    )

    plt.hlines(
        cdf_x1,
        0,
        x1,
        linestyles="--",
        linewidth=1.5,
    )

    plt.hlines(
        cdf_x2,
        0,
        x2,
        linestyles="--",
        linewidth=1.5,
    )

    plt.annotate(
        f"F(x₁) = {cdf_x1:.4f}",
        xy=(x1, cdf_x1),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=11,
    )

    plt.annotate(
        f"F(x₂) = {cdf_x2:.4f}",
        xy=(x2, cdf_x2),
        xytext=(10, -25),
        textcoords="offset points",
        fontsize=11,
    )

    plt.annotate(
        f"x₁ = {x1:.4f}",
        xy=(x1, 0),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=10,
    )

    plt.annotate(
        f"x₂ = {x2:.4f}",
        xy=(x2, 0),
        xytext=(5, 8),
        textcoords="offset points",
        fontsize=10,
    )

    plt.title(
        "Функция распределения F(x) "
        f"с границами областей: k={k}, θ={theta}, N={N}",
        fontsize=15,
    )

    plt.xlabel(
        "x",
        fontsize=13,
    )

    plt.ylabel(
        "F(x)",
        fontsize=13,
    )

    plt.ylim(
        0,
        1.05,
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


def plot_cdf_histogram(
    cdf_area_probabilities,
    k,
    theta,
    N,
):
    """
    Строит вторую гистограмму на основе CDF.

    Высоты столбцов:

        Область 1 = F(x1)
        Область 2 = F(x2) - F(x1)
        Область 3 = 1 - F(x2)

    В отличие от replay-гистограммы, здесь используются
    теоретические вероятности, рассчитанные непосредственно
    через F(x).
    """

    figure = plt.figure(
        figsize=(9, 7)
    )

    values = [
        cdf_area_probabilities["orange_probability"],
        cdf_area_probabilities["blue_probability"],
        cdf_area_probabilities["green_probability"],
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
        "Теоретическая вероятность",
        fontsize=13,
    )

    plt.xlabel(
        "Область",
        fontsize=13,
    )

    plt.title(
        "Теоретическое распределение по областям "
        f"через F(x): k={k}, θ={theta}, N={N}",
        fontsize=15,
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    return figure
