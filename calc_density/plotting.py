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


def plot_replays_histogram(replay_results, N, k, theta):
    """Строит один рисунок со столбцами всех replay.

    Для каждого replay отображаются три столбца: оранжевая, синяя
    и зелёная области. Высота каждого столбца равна count / N,
    а подпись показывает фактическое количество точек.
    Функция только строит и возвращает figure, не сохраняет файлы.
    """
    if N <= 0:
        raise ValueError("N должно быть > 0")
    if not replay_results:
        raise ValueError("Список replay_results пуст")

    replay_numbers = [int(result["replay"]) for result in replay_results]
    orange = [int(result["orange_count"]) for result in replay_results]
    blue = [int(result["blue_count"]) for result in replay_results]
    green = [int(result["green_count"]) for result in replay_results]

    for replay_number, counts in zip(replay_numbers, zip(orange, blue, green)):
        if sum(counts) != N:
            raise ValueError(
                f"Сумма точек replay {replay_number} не равна N: "
                f"{sum(counts)} != {N}"
            )

    import numpy as np

    x = np.arange(len(replay_results))
    width = 0.24
    series = [
        (orange, "Оранжевая", "#E99A37", -width),
        (blue, "Синяя", "#438BD0", 0),
        (green, "Зелёная", "#42966D", width),
    ]

    figure_width = max(11, len(replay_results) * 1.1)
    figure, ax = plt.subplots(figsize=(figure_width, 7))

    for counts, label, color, offset in series:
        ratios = [count / N for count in counts]
        bars = ax.bar(x + offset, ratios, width=width, label=label, color=color)
        for bar, count, ratio in zip(bars, counts, ratios):
            y = ratio / 2 if ratio > 0 else 0.012
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                str(count),
                ha="center",
                va="center" if ratio > 0 else "bottom",
                fontsize=9,
                fontweight="bold",
                color="white" if ratio >= 0.12 else "black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(number) for number in replay_numbers])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Номер replay", fontsize=13)
    ax.set_ylabel("Доля точек n / N", fontsize=13)
    ax.set_title(
        f"Распределение точек по областям для всех replay: "
        f"k={k}, θ={theta}, N={N}",
        fontsize=14,
    )
    ax.legend(title="Область")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.tight_layout()
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


def plot_equal_cdf(x, equal_cdf_areas, k, theta, N):
    """CDF с границами метода 2: F(x)=1/3 и F(x)=2/3."""
    figure = plt.figure(figsize=(11, 7))
    x1 = equal_cdf_areas["orange_boundary"]
    x2 = equal_cdf_areas["green_boundary"]
    y1 = equal_cdf_areas["cdf_orange"]
    y2 = equal_cdf_areas["cdf_green"]
    plt.plot(x, equal_cdf_areas["cdf_values"], linewidth=2,
             label=r"$F(x)$ — функция распределения")
    plt.scatter([x1, x2], [y1, y2], s=55, zorder=5,
                label="Границы равных областей по F(x)")
    for boundary, level in ((x1, y1), (x2, y2)):
        plt.vlines(boundary, 0, level, linestyles="--", linewidth=1.5)
        plt.hlines(level, 0, boundary, linestyles="--", linewidth=1.5)
    plt.annotate(f"F(x₁) = {y1:.4f}", xy=(x1, y1), xytext=(10, 10),
                 textcoords="offset points", fontsize=11)
    plt.annotate(f"F(x₂) = {y2:.4f}", xy=(x2, y2), xytext=(10, -25),
                 textcoords="offset points", fontsize=11)
    plt.annotate(f"x₁ = {x1:.4f}", xy=(x1, 0), xytext=(5, 8),
                 textcoords="offset points", fontsize=10)
    plt.annotate(f"x₂ = {x2:.4f}", xy=(x2, 0), xytext=(5, 8),
                 textcoords="offset points", fontsize=10)
    plt.title(f"CDF: три равные области по F(x): k={k}, θ={theta}, N={N}", fontsize=15)
    plt.xlabel("x", fontsize=13)
    plt.ylabel("F(x)", fontsize=13)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=11)
    plt.tight_layout()
    return figure


def plot_equal_cdf_histogram(equal_cdf_areas, k, theta, N):
    """Теоретическая гистограмма вероятностей метода 2."""
    figure = plt.figure(figsize=(9, 7))
    values = [equal_cdf_areas["orange_probability"],
              equal_cdf_areas["blue_probability"],
              equal_cdf_areas["green_probability"]]
    labels = ["Область 1", "Область 2", "Область 3"]
    bars = plt.bar(labels, values, width=0.6)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                 ha="center", va="bottom", fontsize=11)
    plt.ylim(0, 1.05)
    plt.ylabel("Теоретическая вероятность", fontsize=13)
    plt.xlabel("Область", fontsize=13)
    plt.title(f"Теоретические вероятности метода 2: k={k}, θ={theta}, N={N}", fontsize=15)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return figure


def plot_method2_replays_histogram(replay_results, N, k, theta):
    """Строит единый рисунок с гистограммами всех replay для метода 2."""
    if N <= 0:
        raise ValueError("N должно быть > 0")
    if not replay_results:
        raise ValueError("Список replay_results пуст")

    replay_numbers = [int(result["replay"]) for result in replay_results]
    keys = ("method2_orange_count", "method2_blue_count", "method2_green_count")
    labels = ("Оранжевая", "Синяя", "Зелёная")
    colors = ("#E99A37", "#438BD0", "#42966D")
    counts_by_area = [[int(result[key]) for result in replay_results] for key in keys]

    for number, result, counts in zip(replay_numbers, replay_results, zip(*counts_by_area)):
        if any(key not in result for key in keys):
            raise ValueError("В replay_results отсутствуют счётчики метода 2")
        if sum(counts) != N:
            raise ValueError(
                f"Сумма точек метода 2 для replay {number} не равна N: "
                f"{sum(counts)} != {N}"
            )

    import numpy as np
    x = np.arange(len(replay_results))
    width = 0.24
    figure, ax = plt.subplots(figsize=(max(11, len(replay_results) * 1.1), 7))

    for counts, label, color, offset in zip(
        counts_by_area, labels, colors, (-width, 0, width)
    ):
        ratios = [count / N for count in counts]
        bars = ax.bar(x + offset, ratios, width=width, label=label, color=color)
        for bar, count, ratio in zip(bars, counts, ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ratio / 2 if ratio > 0 else 0.012,
                str(count), ha="center",
                va="center" if ratio > 0 else "bottom",
                fontsize=9, fontweight="bold",
                color="white" if ratio >= 0.12 else "black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(number) for number in replay_numbers])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Номер replay", fontsize=13)
    ax.set_ylabel("Доля точек n / N", fontsize=13)
    ax.set_title(
        f"Метод 2: распределение точек по областям для всех replay: "
        f"k={k}, θ={theta}, N={N}", fontsize=14,
    )
    ax.legend(title="Область")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_method3_distribution(x, f, f_min, f_max, method3_areas, k, theta, N):
    """Строит исходную плотность с делением диапазона x на три равные части.

    Функция только создаёт и возвращает figure; на диск ничего не сохраняет.
    """
    import numpy as np

    x_values = np.asarray(x, dtype=float)
    density = np.asarray(f, dtype=float)
    x1 = float(method3_areas["orange_boundary"])
    x2 = float(method3_areas["green_boundary"])

    figure, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x_values, density, linewidth=2, label=r"$f(x)$ — исходная плотность")
    ax.plot(x_values, np.asarray(f_min, dtype=float), linewidth=1.6, label=r"$f_{\min}(x)$ — минимум")
    ax.plot(x_values, np.asarray(f_max, dtype=float), linewidth=1.6, label=r"$f_{\max}(x)$ — максимум")
    ax.fill_between(
        x_values, 0, density, where=(x_values <= x1),
        color="#E99A37", alpha=0.28, label="Область 1",
    )
    ax.fill_between(
        x_values, 0, density, where=((x_values > x1) & (x_values <= x2)),
        color="#438BD0", alpha=0.28, label="Область 2",
    )
    ax.fill_between(
        x_values, 0, density, where=(x_values > x2),
        color="#42966D", alpha=0.28, label="Область 3",
    )
    ax.axvline(x1, color="#B87521", linestyle="--", linewidth=1.6)
    ax.axvline(x2, color="#286DA8", linestyle="--", linewidth=1.6)
    ax.annotate(f"x₁ = {x1:.4f}", xy=(x1, 0), xytext=(5, 10),
                textcoords="offset points", fontsize=10)
    ax.annotate(f"x₂ = {x2:.4f}", xy=(x2, 0), xytext=(5, 10),
                textcoords="offset points", fontsize=10)
    ax.set_xlim(float(method3_areas["x_min"]), float(method3_areas["x_max"]))
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("Плотность", fontsize=13)
    ax.text(0.5, 0.96, "Равные интервалы по x", transform=ax.transAxes, ha="center", va="top", fontsize=11, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    ax.set_title(
        f"Метод 3: равные интервалы по x: k={k}, θ={theta}, N={N}",
        fontsize=15,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    figure.tight_layout()
    return figure


def plot_method3_cdf(x, method3_areas, k, theta, N):
    """Строит CDF метода 3 и отмечает две точки (x₁, F(x₁)) и (x₂, F(x₂)).

    Координаты точек используют те же границы x, что и график плотности.
    """
    x1 = float(method3_areas["orange_boundary"])
    x2 = float(method3_areas["green_boundary"])
    y1 = float(method3_areas["cdf_orange"])
    y2 = float(method3_areas["cdf_green"])
    cdf_values = method3_areas["cdf_values"]

    figure, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x, cdf_values, linewidth=2, label=r"$F(x)$ — функция распределения")

    # Ровно две точки границ областей — лежат на кривой CDF.
    ax.scatter(
        [x1, x2], [y1, y2], s=65, zorder=5,
        color="#333333", label="Точки границ областей",
    )
    for boundary, level in ((x1, y1), (x2, y2)):
        ax.vlines(boundary, 0, level, linestyles="--", linewidth=1.5, alpha=0.8)
        ax.hlines(level, float(method3_areas["x_min"]), boundary,
                  linestyles="--", linewidth=1.2, alpha=0.8)
    ax.annotate(f"(x₁, F(x₁))\n({x1:.4f}, {y1:.4f})",
                xy=(x1, y1), xytext=(10, 12), textcoords="offset points", fontsize=10)
    ax.annotate(f"(x₂, F(x₂))\n({x2:.4f}, {y2:.4f})",
                xy=(x2, y2), xytext=(10, -34), textcoords="offset points", fontsize=10)
    ax.set_xlim(float(method3_areas["x_min"]), float(method3_areas["x_max"]))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("F(x)", fontsize=13)
    ax.set_title(
        f"Метод 3: CDF с границами равных интервалов x: k={k}, θ={theta}, N={N}",
        fontsize=15,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    figure.tight_layout()
    return figure


def plot_method3_theoretical_histogram(method3_areas, k, theta, N):
    """Строит теоретическую гистограмму вероятностей метода по равным интервалам x."""
    values = [
        float(method3_areas["orange_probability"]),
        float(method3_areas["blue_probability"]),
        float(method3_areas["green_probability"]),
    ]
    labels = ["Область 1", "Область 2", "Область 3"]
    colors = ["#E99A37", "#438BD0", "#42966D"]

    figure, ax = plt.subplots(figsize=(9, 7))
    bars = ax.bar(labels, values, width=0.6, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value,
            f"{value:.4f}", ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Теоретическая вероятность", fontsize=13)
    ax.set_xlabel("Область", fontsize=13)
    ax.set_title(
        f"Метод 3: теоретические вероятности по равным интервалам x: "
        f"k={k}, θ={theta}, N={N}",
        fontsize=14,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_method3_replays_histogram(replay_results, N, k, theta):
    """Строит гистограммы областей метода 3 для каждого replay на одном рисунке."""
    import numpy as np

    if N <= 0:
        raise ValueError("N должно быть > 0")
    if not replay_results:
        raise ValueError("Список replay_results пуст")

    replay_numbers = [int(result["replay"]) for result in replay_results]
    keys = ("method3_orange_count", "method3_blue_count", "method3_green_count")
    labels = ("Оранжевая", "Синяя", "Зелёная")
    colors = ("#E99A37", "#438BD0", "#42966D")
    for result, number in zip(replay_results, replay_numbers):
        if any(key not in result for key in keys):
            raise ValueError(f"В replay {number} отсутствуют счётчики метода 4")
        counts = [int(result[key]) for key in keys]
        if sum(counts) != N:
            raise ValueError(
                f"Сумма точек метода 4 для replay {number} не равна N: "
                f"{sum(counts)} != {N}"
            )

    counts_by_area = [[int(result[key]) for result in replay_results] for key in keys]
    positions = np.arange(len(replay_results))
    width = 0.24
    figure, ax = plt.subplots(figsize=(max(11, len(replay_results) * 1.1), 7))

    for counts, label, color, offset in zip(
        counts_by_area, labels, colors, (-width, 0, width)
    ):
        ratios = [count / N for count in counts]
        bars = ax.bar(positions + offset, ratios, width=width, label=label, color=color)
        for bar, count, ratio in zip(bars, counts, ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ratio / 2 if ratio > 0 else 0.012,
                str(count), ha="center",
                va="center" if ratio > 0 else "bottom",
                fontsize=9, fontweight="bold",
                color="white" if ratio >= 0.12 else "black",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(number) for number in replay_numbers])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Номер replay", fontsize=13)
    ax.set_ylabel("Доля точек n / N", fontsize=13)
    ax.set_title(
        f"Метод 3: распределение точек по равным интервалам x для всех replay: "
        f"k={k}, θ={theta}, N={N}", fontsize=14,
    )
    ax.legend(title="Область")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.tight_layout()
    return figure
