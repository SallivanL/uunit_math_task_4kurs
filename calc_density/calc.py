import numpy as np
from scipy.stats import gamma


def calculate_densities(k, theta, N, points=5000):
    """
    Рассчитывает:

        f      — исходное распределение Gamma(k, theta)
        f_min  — распределение минимума выборки размера N
        f_max  — распределение максимума выборки размера N
    """

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
        scale=theta,
    )

    x = np.linspace(
        0,
        x_max,
        points,
    )

    f = gamma.pdf(
        x,
        a=k,
        scale=theta,
    )

    F = gamma.cdf(
        x,
        a=k,
        scale=theta,
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


def find_left_area(x, y, m):
    """
    Находит левую область, содержащую m долю площади
    под кривой y.

    Используется для f_min.

    Результат:
        [0, right]
    """

    if not 0 < m <= 1:
        raise ValueError("m должно быть в диапазоне (0, 1]")

    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    right = len(x) - 1

    for i in range(1, len(x)):
        area = np.trapezoid(
            y[:i + 1],
            x[:i + 1],
        )

        if area >= target_area:
            right = i
            break

    area = np.trapezoid(
        y[:right + 1],
        x[:right + 1],
    )

    return {
        "left": 0,
        "right": right,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area,
    }


def find_right_area(x, y, m):
    """
    Находит правую область, содержащую m долю площади
    под кривой y.

    Используется для f_max.

    Результат:
        [left, x_max]
    """

    if not 0 < m <= 1:
        raise ValueError("m должно быть в диапазоне (0, 1]")

    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    left = 0

    for i in range(len(x) - 2, -1, -1):
        area = np.trapezoid(
            y[i:],
            x[i:],
        )

        if area >= target_area:
            left = i
            break

    area = np.trapezoid(
        y[left:],
        x[left:],
    )

    return {
        "left": left,
        "right": len(x) - 1,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area,
    }


def find_middle_area(x, left_index, right_index):
    """
    Формирует центральную область между двумя границами.

    Левая граница берётся из f_min.
    Правая граница берётся из f_max.
    """

    if left_index >= right_index:
        raise ValueError(
            "Левая граница должна быть меньше правой"
        )

    return {
        "left": left_index,
        "right": right_index,
        "left_boundary": x[left_index],
        "right_boundary": x[right_index],
    }


def calculate_areas(
    x,
    f,
    f_min,
    f_max,
    m,
):
    """
    Формирует три НЕпересекающиеся области.

    1. Оранжевая:
       левая область f_min, содержащая m площади.

    2. Синяя:
       промежуток между правой границей оранжевой
       и левой границей зелёной.

    3. Зелёная:
       правая область f_max, содержащая m площади.

    Всё пространство x разбивается на:

        ОРАНЖЕВАЯ | СИНЯЯ | ЗЕЛЁНАЯ
    """

    orange = find_left_area(
        x=x,
        y=f_min,
        m=m,
    )

    green = find_right_area(
        x=x,
        y=f_max,
        m=m,
    )

    blue = find_middle_area(
        x=x,
        left_index=orange["right"],
        right_index=green["left"],
    )

    return {
        "orange": orange,
        "blue": blue,
        "green": green,
    }


def generate_sample(k, theta, N, rng):
    """
    Генерирует одну выборку из N элементов Gamma(k, theta).
    """

    if k <= 0:
        raise ValueError("k должно быть > 0")

    if theta <= 0:
        raise ValueError("theta должно быть > 0")

    if N <= 0:
        raise ValueError("N должно быть > 0")

    return rng.gamma(
        shape=k,
        scale=theta,
        size=int(N),
    )


def calculate_replay_statistics(
    sample,
    x,
    areas,
):
    """
    Классифицирует каждый элемент выборки
    ровно в одну из трёх областей.

    Оранжевая:
        x <= orange_boundary

    Синяя:
        orange_boundary < x < green_boundary

    Зелёная:
        x >= green_boundary
    """

    orange_boundary = x[
        areas["orange"]["right"]
    ]

    green_boundary = x[
        areas["green"]["left"]
    ]

    orange_mask = (
        sample <= orange_boundary
    )

    blue_mask = (
        (sample > orange_boundary)
        & (sample < green_boundary)
    )

    green_mask = (
        sample >= green_boundary
    )

    orange_count = int(
        np.sum(orange_mask)
    )

    blue_count = int(
        np.sum(blue_mask)
    )

    green_count = int(
        np.sum(green_mask)
    )

    total_count = len(sample)

    covered_count = (
        orange_count
        + blue_count
        + green_count
    )

    if covered_count != total_count:
        raise RuntimeError(
            "Не удалось классифицировать все элементы "
            "выборки."
        )

    return {
        "orange": {
            "count": orange_count,
            "ratio": orange_count / total_count,
        },
        "blue": {
            "count": blue_count,
            "ratio": blue_count / total_count,
        },
        "green": {
            "count": green_count,
            "ratio": green_count / total_count,
        },
        "total": total_count,
        "boundaries": {
            "orange": orange_boundary,
            "green": green_boundary,
        },
    }


def run_replays(
    k,
    theta,
    N,
    replays,
    x,
    areas,
    rng,
):
    """
    Выполняет replays независимых экспериментов.
    """

    if replays <= 0:
        raise ValueError("replays должно быть > 0")

    results = []

    for replay in range(1, replays + 1):
        sample = generate_sample(
            k=k,
            theta=theta,
            N=N,
            rng=rng,
        )

        statistics = calculate_replay_statistics(
            sample=sample,
            x=x,
            areas=areas,
        )

        results.append({
            "replay": replay,
            "orange_count": statistics["orange"]["count"],
            "blue_count": statistics["blue"]["count"],
            "green_count": statistics["green"]["count"],
            "total": statistics["total"],
        })

    return results


def calculate_replay_ratios(replay_results):
    """
    Рассчитывает n / N для каждой области
    отдельно для каждого replay.

    Для каждого replay:

        orange_ratio = n1 / N
        blue_ratio   = n2 / N
        green_ratio  = n3 / N

    Сумма трёх долей для каждого replay должна быть равна 1.
    """

    if not replay_results:
        raise ValueError(
            "replay_results не должен быть пустым"
        )

    ratios = []

    for result in replay_results:
        total = result["total"]

        if total <= 0:
            raise ValueError(
                "Количество элементов replay должно быть > 0"
            )

        orange_ratio = (
            result["orange_count"] / total
        )

        blue_ratio = (
            result["blue_count"] / total
        )

        green_ratio = (
            result["green_count"] / total
        )

        ratio_sum = (
            orange_ratio
            + blue_ratio
            + green_ratio
        )

        if not np.isclose(
            ratio_sum,
            1.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Сумма долей областей для replay "
                "не равна 1."
            )

        ratios.append({
            "replay": result["replay"],
            "orange_ratio": float(orange_ratio),
            "blue_ratio": float(blue_ratio),
            "green_ratio": float(green_ratio),
        })

    return ratios


def calculate_average_replay_ratios(replay_ratios):
    """
    Рассчитывает среднюю долю элементов
    в каждой области по всем replay.

    Результат:

        mean(n1 / N)
        mean(n2 / N)
        mean(n3 / N)
    """

    if not replay_ratios:
        raise ValueError(
            "replay_ratios не должен быть пустым"
        )

    orange_ratio = np.mean([
        replay["orange_ratio"]
        for replay in replay_ratios
    ])

    blue_ratio = np.mean([
        replay["blue_ratio"]
        for replay in replay_ratios
    ])

    green_ratio = np.mean([
        replay["green_ratio"]
        for replay in replay_ratios
    ])

    ratio_sum = (
        orange_ratio
        + blue_ratio
        + green_ratio
    )

    if not np.isclose(
        ratio_sum,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Средние доли трёх областей "
            "не дают сумму 1."
        )

    return {
        "orange_ratio": float(orange_ratio),
        "blue_ratio": float(blue_ratio),
        "green_ratio": float(green_ratio),
        "total": float(ratio_sum),
    }


def calculate_replay_totals(replay_results):
    """
    Суммирует результаты всех replay.
    """

    total = sum(
        result["total"]
        for result in replay_results
    )

    orange_count = sum(
        result["orange_count"]
        for result in replay_results
    )

    blue_count = sum(
        result["blue_count"]
        for result in replay_results
    )

    green_count = sum(
        result["green_count"]
        for result in replay_results
    )

    covered_count = (
        orange_count
        + blue_count
        + green_count
    )

    if covered_count != total:
        raise RuntimeError(
            "Сумма трёх областей не равна общему "
            "количеству элементов."
        )

    return {
        "orange_count": orange_count,
        "orange_ratio": orange_count / total,
        "blue_count": blue_count,
        "blue_ratio": blue_count / total,
        "green_count": green_count,
        "green_ratio": green_count / total,
        "total": total,
    }


def calculate_area_statistics(
    k,
    theta,
    N,
    x,
    areas,
    replay_results,
):
    """
    Рассчитывает статистику трёх областей.

    Границы определяются только по f_min и f_max.

    Оранжевая:
        X <= orange_boundary

    Синяя:
        orange_boundary < X < green_boundary

    Зелёная:
        X >= green_boundary
    """

    orange_boundary = x[
        areas["orange"]["right"]
    ]

    green_boundary = x[
        areas["green"]["left"]
    ]

    p_orange = gamma.cdf(
        orange_boundary,
        a=k,
        scale=theta,
    )

    p_green_start = gamma.cdf(
        green_boundary,
        a=k,
        scale=theta,
    )

    p_blue = (
        p_green_start
        - p_orange
    )

    p_green = (
        1
        - p_green_start
    )

    theoretical_probabilities = {
        "orange": p_orange,
        "blue": p_blue,
        "green": p_green,
    }

    probability_sum = (
        p_orange
        + p_blue
        + p_green
    )

    if not np.isclose(
        probability_sum,
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Теоретические вероятности трёх областей "
            "не дают сумму 1."
        )

    statistics = {}

    for area in [
        "orange",
        "blue",
        "green",
    ]:
        counts = np.array(
            [
                replay[f"{area}_count"]
                for replay in replay_results
            ],
            dtype=float,
        )

        total_elements = np.sum(
            counts
        )

        total_replay_elements = (
            len(replay_results)
            * int(N)
        )

        probability = (
            theoretical_probabilities[area]
        )

        mathematical_expectation = (
            N * probability
        )

        frequency = (
            total_elements
            / total_replay_elements
        )

        mean = np.mean(
            counts
        )

        variance = np.var(
            counts,
            ddof=0,
        )

        standard_deviation = np.sqrt(
            variance
        )

        statistics[area] = {
            "mathematical_expectation": float(
                mathematical_expectation
            ),
            "frequency": float(
                frequency
            ),
            "mean": float(
                mean
            ),
            "standard_deviation": float(
                standard_deviation
            ),
            "variance": float(
                variance
            ),
        }

    return statistics