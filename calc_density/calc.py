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


def calculate_cdf_area_probabilities(
    k,
    theta,
    x,
    areas,
):
    """
    Рассчитывает теоретические вероятности трёх областей
    через функцию распределения F(x) исходного Gamma(k, theta).

    x1 = правая граница оранжевой области.
    x2 = левая граница зелёной области.

    Тогда:

        P1 = F(x1)
        P2 = F(x2) - F(x1)
        P3 = 1 - F(x2)

    Эти значения являются теоретическими долями элементов
    в трёх областях.
    """

    if k <= 0:
        raise ValueError("k должно быть > 0")

    if theta <= 0:
        raise ValueError("theta должно быть > 0")

    if len(x) == 0:
        raise ValueError("x не должен быть пустым")

    orange_boundary = x[
        areas["orange"]["right"]
    ]

    green_boundary = x[
        areas["green"]["left"]
    ]

    if orange_boundary >= green_boundary:
        raise ValueError(
            "Граница оранжевой области должна быть меньше "
            "границы зелёной области"
        )

    cdf_orange = gamma.cdf(
        orange_boundary,
        a=k,
        scale=theta,
    )

    cdf_green = gamma.cdf(
        green_boundary,
        a=k,
        scale=theta,
    )

    orange_probability = cdf_orange

    blue_probability = (
        cdf_green
        - cdf_orange
    )

    green_probability = (
        1.0
        - cdf_green
    )

    probability_sum = (
        orange_probability
        + blue_probability
        + green_probability
    )

    if not np.isclose(
        probability_sum,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Теоретические вероятности через F(x) "
            "не дают сумму 1."
        )

    cdf_values = gamma.cdf(
        x,
        a=k,
        scale=theta,
    )

    return {
        "orange_boundary": float(
            orange_boundary
        ),
        "green_boundary": float(
            green_boundary
        ),
        "cdf_orange": float(cdf_orange),
        "cdf_green": float(cdf_green),
        "cdf_values": cdf_values,
        "orange_probability": float(
            orange_probability
        ),
        "blue_probability": float(
            blue_probability
        ),
        "green_probability": float(
            green_probability
        ),
        "total": float(probability_sum),
    }



def calculate_equal_cdf_areas(k, theta, x):
    """Границы метода 2: деление диапазона CDF на три равные части."""
    if k <= 0:
        raise ValueError("k должно быть > 0")
    if theta <= 0:
        raise ValueError("theta должно быть > 0")
    if len(x) == 0:
        raise ValueError("x не должен быть пустым")

    from scipy.stats import gamma

    levels = (1.0 / 3.0, 2.0 / 3.0)
    x1, x2 = gamma.ppf(levels, a=k, scale=theta)
    cdf_values = gamma.cdf(x, a=k, scale=theta)
    probabilities = {
        "orange_probability": float(levels[0]),
        "blue_probability": float(levels[1] - levels[0]),
        "green_probability": float(1.0 - levels[1]),
        "total": 1.0,
    }
    return {
        "orange_boundary": float(x1),
        "green_boundary": float(x2),
        "cdf_orange": float(levels[0]),
        "cdf_green": float(levels[1]),
        "cdf_values": cdf_values,
        **probabilities,
    }



def calculate_method3_equal_x_areas(k, theta, x):
    """Метод 3: делит диапазон x графика на три равных по длине интервала.

    Файлы графиков этого метода используют префикс method_3.
    Теоретические вероятности рассчитываются через исходную CDF Gamma.
    """
    if k <= 0:
        raise ValueError("k должно быть > 0")
    if theta <= 0:
        raise ValueError("theta должно быть > 0")

    x_values = np.asarray(x, dtype=float)
    if x_values.ndim != 1 or x_values.size < 2:
        raise ValueError("x должен содержать как минимум две точки")
    if not np.all(np.isfinite(x_values)):
        raise ValueError("Все значения x должны быть конечными")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("Значения x должны строго возрастать")

    x_min = float(x_values[0])
    x_max = float(x_values[-1])
    if x_max <= x_min:
        raise ValueError("Верхняя граница x должна быть больше нижней")

    interval_width = (x_max - x_min) / 3.0
    boundary_1 = x_min + interval_width
    boundary_2 = x_min + 2.0 * interval_width

    cdf_1 = float(gamma.cdf(boundary_1, a=k, scale=theta))
    cdf_2 = float(gamma.cdf(boundary_2, a=k, scale=theta))
    probabilities = {
        "orange_probability": cdf_1,
        "blue_probability": cdf_2 - cdf_1,
        "green_probability": 1.0 - cdf_2,
    }
    probability_sum = sum(probabilities.values())

    if any(not np.isfinite(value) or value < 0 for value in probabilities.values()):
        raise RuntimeError("Теоретические вероятности метода по равным интервалам x некорректны")
    if not np.isclose(probability_sum, 1.0, atol=1e-12):
        raise RuntimeError("Теоретические вероятности метода по равным интервалам x не дают сумму 1")

    return {
        "x_min": x_min,
        "x_max": x_max,
        "orange_boundary": float(boundary_1),
        "green_boundary": float(boundary_2),
        "cdf_orange": cdf_1,
        "cdf_green": cdf_2,
        "cdf_values": gamma.cdf(x_values, a=k, scale=theta),
        **probabilities,
        "total": float(probability_sum),
    }


def calculate_method3_replay_totals(replay_results):
    """Суммирует фактические количества областей метода с префиксом method3."""
    if not replay_results:
        raise ValueError("replay_results не должен быть пустым")

    keys = ("method3_orange_count", "method3_blue_count", "method3_green_count")
    if any(any(key not in result for key in keys) for result in replay_results):
        raise ValueError("В replay_results отсутствуют счётчики метода 4")

    counts = [sum(int(result[key]) for result in replay_results) for key in keys]
    total = sum(int(result["total"]) for result in replay_results)
    if total <= 0:
        raise ValueError("Общее количество элементов должно быть больше нуля")
    if sum(counts) != total:
        raise RuntimeError("Сумма областей метода 4 не равна общему количеству элементов")

    return {
        "orange_count": counts[0],
        "blue_count": counts[1],
        "green_count": counts[2],
        "orange_ratio": counts[0] / total,
        "blue_ratio": counts[1] / total,
        "green_ratio": counts[2] / total,
        "total": total,
    }

def calculate_method2_replay_totals(replay_results):
    """Суммарные фактические количества точек для метода 2."""
    if not replay_results:
        raise ValueError("replay_results не должен быть пустым")
    keys = ("method2_orange_count", "method2_blue_count", "method2_green_count")
    counts = [sum(int(r[key]) for r in replay_results) for key in keys]
    total = sum(int(r["total"]) for r in replay_results)
    if sum(counts) != total:
        raise RuntimeError("Сумма областей метода 2 не равна общему количеству элементов.")
    return {
        "orange_count": counts[0], "blue_count": counts[1], "green_count": counts[2],
        "orange_ratio": counts[0] / total, "blue_ratio": counts[1] / total,
        "green_ratio": counts[2] / total, "total": total,
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
    equal_cdf_areas=None,
    method3_areas=None,
):
    """
    Выполняет replay; все методы используют одну и ту же выборку.
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

        method2_counts = {}
        if equal_cdf_areas is not None:
            b1 = equal_cdf_areas["orange_boundary"]
            b2 = equal_cdf_areas["green_boundary"]
            method2_counts = {
                "method2_orange_count": int(np.sum(sample <= b1)),
                "method2_blue_count": int(np.sum((sample > b1) & (sample < b2))),
                "method2_green_count": int(np.sum(sample >= b2)),
            }
            if sum(method2_counts.values()) != len(sample):
                raise RuntimeError("Метод 2 не классифицировал все элементы выборки.")

        method3_counts = {}
        if method3_areas is not None:
            b1 = method3_areas["orange_boundary"]
            b2 = method3_areas["green_boundary"]
            method3_counts = {
                "method3_orange_count": int(np.sum(sample <= b1)),
                "method3_blue_count": int(np.sum((sample > b1) & (sample < b2))),
                "method3_green_count": int(np.sum(sample >= b2)),
            }
            if sum(method3_counts.values()) != len(sample):
                raise RuntimeError("Метод 4 не классифицировал все элементы выборки")

        results.append({
            "replay": replay,
            "orange_count": statistics["orange"]["count"],
            "blue_count": statistics["blue"]["count"],
            "green_count": statistics["green"]["count"],
            **method2_counts,
            **method3_counts,
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

def calculate_chi_squared_comparison(
    theoretical_probabilities,
    observed_counts,
):
    """
    Сравнивает теоретические и экспериментальные вероятности
    трёх областей по заданной формуле:

        chi2 = sum(((p_j - p_hat_j) / p_j) ** 2)

    theoretical_probabilities и observed_counts должны содержать
    ключи orange, blue и green. Фактические вероятности
    рассчитываются по общему количеству элементов всех replay.
    """
    area_names = ("orange", "blue", "green")

    missing_probabilities = [
        area for area in area_names
        if area not in theoretical_probabilities
    ]
    missing_counts = [
        area for area in area_names
        if area not in observed_counts
    ]
    if missing_probabilities:
        raise ValueError(
            "Не заданы теоретические вероятности областей: "
            + ", ".join(missing_probabilities)
        )
    if missing_counts:
        raise ValueError(
            "Не заданы фактические количества областей: "
            + ", ".join(missing_counts)
        )

    probabilities = {
        area: float(theoretical_probabilities[area])
        for area in area_names
    }
    counts = {
        area: int(observed_counts[area])
        for area in area_names
    }

    if any(not np.isfinite(p) or p <= 0 for p in probabilities.values()):
        raise ValueError(
            "Для расчёта chi² все теоретические вероятности "
            "должны быть конечными и строго больше нуля."
        )
    if any(count < 0 for count in counts.values()):
        raise ValueError("Фактические количества не могут быть отрицательными.")

    probability_sum = sum(probabilities.values())
    if not np.isclose(probability_sum, 1.0, atol=1e-10):
        raise ValueError(
            "Сумма теоретических вероятностей трёх областей "
            f"должна быть равна 1, получено {probability_sum:.12g}."
        )

    total_count = sum(counts.values())
    if total_count <= 0:
        raise ValueError(
            "Общее фактическое количество элементов должно быть больше нуля."
        )

    areas = {}
    chi_squared = 0.0
    for area in area_names:
        p = probabilities[area]
        p_hat = counts[area] / total_count
        contribution = ((p - p_hat) / p) ** 2
        areas[area] = {
            "theoretical_probability": p,
            "empirical_probability": float(p_hat),
            "observed_count": counts[area],
            "total_count": total_count,
            "contribution": float(contribution),
        }
        chi_squared += contribution

    empirical_sum = sum(
        result["empirical_probability"] for result in areas.values()
    )
    if not np.isclose(empirical_sum, 1.0, atol=1e-12):
        raise RuntimeError(
            "Сумма экспериментальных вероятностей трёх областей не равна 1."
        )

    return {
        "areas": areas,
        "chi_squared": float(chi_squared),
        "total_count": total_count,
    }



def calculate_expected_squared_error(theoretical_probabilities, sample_size):
    """Вычисляет M[X²] для ошибки эмпирических долей по трём областям.

    Для каждой области X_j = p_hat_j - p_j, где p_hat_j — эмпирическая
    доля в выборке размера N, а p_j — теоретическая вероятность.
    На график передаётся M[X²] для X² = sum_j (p_hat_j - p_j)²:

        M[X²] = sum_j p_j * (1 - p_j) / N.

    Это математическое ожидание квадрата абсолютной ошибки, а не
    нормированная статистика chi-squared. Значение вычисляется аналитически,
    поэтому кривая не зависит от случайного шума отдельных replay.
    """
    if not isinstance(sample_size, (int, np.integer)) or sample_size <= 0:
        raise ValueError("sample_size должно быть положительным целым числом")

    area_names = ("orange", "blue", "green")
    missing = [area for area in area_names if area not in theoretical_probabilities]
    if missing:
        raise ValueError(
            "Не заданы теоретические вероятности областей: " + ", ".join(missing)
        )

    probabilities = np.asarray(
        [float(theoretical_probabilities[area]) for area in area_names],
        dtype=float,
    )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Теоретические вероятности должны быть конечными")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Теоретические вероятности должны находиться в [0, 1]")
    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-10):
        raise ValueError("Сумма теоретических вероятностей должна быть равна 1")

    expected_squared_error = np.sum(probabilities * (1.0 - probabilities)) / int(sample_size)
    return float(expected_squared_error)



def calculate_mean_chi_squared_by_replay(
    replay_results,
    theoretical_probabilities,
    count_keys,
):
    """Рассчитывает M[χ²] как среднее значение χ² по отдельным replay.

    Для каждого replay вычисляется та же статистика, что и в
    calculate_chi_squared_comparison, но по счётчикам только этого replay.
    Возвращаются среднее значение и список значений по всем повторам.

    count_keys — соответствие областей именам счётчиков в replay_results,
    например {"orange": "orange_count", ...}.
    """
    if not replay_results:
        raise ValueError("replay_results не должен быть пустым")

    area_names = ("orange", "blue", "green")
    missing_areas = [area for area in area_names if area not in count_keys]
    if missing_areas:
        raise ValueError(
            "Не заданы ключи счётчиков областей: " + ", ".join(missing_areas)
        )

    chi_squared_values = []
    for replay in replay_results:
        missing_counts = [
            count_keys[area]
            for area in area_names
            if count_keys[area] not in replay
        ]
        if missing_counts:
            replay_number = replay.get("replay", "неизвестен")
            raise ValueError(
                f"В replay {replay_number} отсутствуют счётчики: "
                + ", ".join(missing_counts)
            )

        observed_counts = {
            area: int(replay[count_keys[area]])
            for area in area_names
        }
        comparison = calculate_chi_squared_comparison(
            theoretical_probabilities=theoretical_probabilities,
            observed_counts=observed_counts,
        )
        chi_squared_values.append(comparison["chi_squared"])

    return {
        "mean": float(np.mean(chi_squared_values)),
        "values": chi_squared_values,
    }
