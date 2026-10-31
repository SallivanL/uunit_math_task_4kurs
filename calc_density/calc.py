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
    theoretical_probabilities=None,
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

    if theoretical_probabilities is None:
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

        p_blue = p_green_start - p_orange
        p_green = 1.0 - p_green_start

        theoretical_probabilities = {
            "orange": p_orange,
            "blue": p_blue,
            "green": p_green,
        }

    probability_sum = sum(
        float(theoretical_probabilities[area])
        for area in ("orange", "blue", "green")
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


# -----------------------------------------------------------------------------
# Смешанная выборка из нескольких Gamma-распределений.
# Эти функции используются только когда massive_k содержит более одного k.
# -----------------------------------------------------------------------------


def calculate_mixed_sample_counts(k_values, N):
    """Распределяет N элементов между k максимально равномерно.

    Остаток распределяется по одному элементу первым компонентам.
    Например, N=1000 и три k дают [334, 333, 333].
    """
    values = list(k_values)
    if not values:
        raise ValueError("k_values не должен быть пустым")
    if len(set(values)) != len(values):
        raise ValueError("Значения k_values не должны повторяться")
    if any(not np.isfinite(k) or k <= 0 for k in values):
        raise ValueError("Все значения k должны быть конечными и больше нуля")
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N должно быть положительным целым числом")
    if N < len(values):
        raise ValueError(
            "N должно быть не меньше количества k, чтобы каждому распределению "
            "достался хотя бы один элемент"
        )

    base_count, remainder = divmod(int(N), len(values))
    return [base_count + (1 if index < remainder else 0) for index in range(len(values))]


def calculate_mixed_densities(k_values, theta, N, sample_counts, points=5000):
    """Рассчитывает плотности смеси и экстремумов фиксированной смешанной выборки.

    Каждая компонента Gamma(k_i, theta) представлена ровно n_i элементами.
    f_min и f_max — плотности минимума и максимума независимых, но
    не одинаково распределённых элементов.
    """
    k_values = list(k_values)
    sample_counts = [int(value) for value in sample_counts]
    if not k_values or len(k_values) != len(sample_counts):
        raise ValueError("Для каждого k должно быть задано количество элементов")
    if theta <= 0 or not np.isfinite(theta):
        raise ValueError("theta должно быть конечным и больше нуля")
    if N <= 0 or int(N) != N:
        raise ValueError("N должно быть положительным целым числом")
    if sum(sample_counts) != int(N) or any(value < 0 for value in sample_counts):
        raise ValueError("Сумма sample_counts должна быть равна N")
    if any(k <= 0 or not np.isfinite(k) for k in k_values):
        raise ValueError("Все значения k должны быть конечными и больше нуля")
    if points < 2:
        raise ValueError("points должно быть не меньше 2")

    active = [(float(k), count) for k, count in zip(k_values, sample_counts) if count > 0]
    if not active:
        raise ValueError("В смешанной выборке должен быть хотя бы один активный компонент")

    # Область x охватывает практически всю массу распределения максимума.
    quantile_probability = 0.9999 ** (1.0 / int(N))
    x_max = max(
        float(gamma.ppf(quantile_probability, a=k, scale=theta))
        for k, _ in active
    )
    if points >= 3000:
        def minimum_cdf_at(value):
            log_survival = 0.0
            for component_k, component_count in active:
                component_cdf = float(gamma.cdf(value, a=component_k, scale=theta))
                if component_cdf >= 1.0:
                    return 1.0
                log_survival += component_count * np.log1p(-component_cdf)
            return float(-np.expm1(log_survival))

        def mixture_cdf_at(value):
            return float(sum(
                (component_count / int(N))
                * gamma.cdf(value, a=component_k, scale=theta)
                for component_k, component_count in active
            ))

        def bisect_quantile(target, cdf_function):
            low, high = 0.0, x_max
            for _ in range(80):
                middle = (low + high) / 2.0
                if cdf_function(middle) < target:
                    low = middle
                else:
                    high = middle
            return (low + high) / 2.0

        # Плотность минимума сосредоточена около нуля, а плотность максимума —
        # в правом хвосте. Неразбитая равномерная сетка теряла бы точность f_min
        # при большом N и широком диапазоне x. Поэтому выделяем отдельные
        # участки сетки для нижнего хвоста, основной массы и верхнего хвоста.
        minimum_focus = bisect_quantile(0.995, minimum_cdf_at)
        mixture_focus = bisect_quantile(0.95, mixture_cdf_at)
        if mixture_focus <= minimum_focus:
            mixture_focus = minimum_focus + 0.7 * (x_max - minimum_focus)

        low_points = max(2, int(points * 0.35))
        middle_points = max(2, int(points * 0.45))
        high_points = int(points) - low_points - middle_points
        if high_points < 2:
            high_points = 2
            middle_points = max(2, int(points) - low_points - high_points)

        x = np.concatenate((
            np.linspace(0.0, minimum_focus, low_points, endpoint=False),
            np.linspace(minimum_focus, mixture_focus, middle_points, endpoint=False),
            np.linspace(mixture_focus, x_max, high_points),
        ))
    else:
        x = np.linspace(0.0, x_max, int(points))

    component_pdfs = []
    component_cdfs = []
    for k, _ in active:
        component_pdfs.append(gamma.pdf(x, a=k, scale=theta))
        component_cdfs.append(gamma.cdf(x, a=k, scale=theta))

    component_pdfs = np.asarray(component_pdfs, dtype=float)
    component_cdfs = np.asarray(component_cdfs, dtype=float)
    active_counts = np.asarray([count for _, count in active], dtype=int)
    weights = active_counts / float(N)

    f = np.sum(weights[:, None] * component_pdfs, axis=0)

    # f_max(x) = sum_i n_i f_i(x) F_i(x)^(n_i-1) * product_{j!=i} F_j(x)^n_j
    # f_min(x) = sum_i n_i f_i(x) (1-F_i(x))^(n_i-1)
    #           * product_{j!=i} (1-F_j(x))^n_j
    f_max = np.zeros_like(x)
    f_min = np.zeros_like(x)
    for index, count in enumerate(active_counts):
        pdf_i = component_pdfs[index]
        cdf_i = component_cdfs[index]
        other_max = np.ones_like(x)
        other_min = np.ones_like(x)
        for other_index, other_count in enumerate(active_counts):
            if other_index == index:
                continue
            other_cdf = component_cdfs[other_index]
            other_max *= np.power(other_cdf, other_count)
            other_min *= np.power(1.0 - other_cdf, other_count)

        f_max += (
            count * pdf_i * np.power(cdf_i, count - 1) * other_max
        )
        f_min += (
            count * pdf_i * np.power(1.0 - cdf_i, count - 1) * other_min
        )

    return x, f, f_min, f_max


def _mixed_cdf_values(k_values, theta, x, sample_counts):
    """Возвращает CDF общей выборки с фиксированными квотами компонентов."""
    k_values = list(k_values)
    sample_counts = np.asarray(sample_counts, dtype=float)
    if len(k_values) != sample_counts.size or sample_counts.sum() <= 0:
        raise ValueError("Некорректные k_values или sample_counts")
    cdf_values = np.zeros_like(np.asarray(x, dtype=float))
    for k, count in zip(k_values, sample_counts):
        cdf_values += (count / sample_counts.sum()) * gamma.cdf(
            x, a=k, scale=theta
        )
    return cdf_values


def calculate_mixed_cdf_area_probabilities(k_values, theta, x, areas, sample_counts):
    """Теоретические вероятности областей для смеси Gamma-компонентов."""
    orange_boundary = float(x[areas["orange"]["right"]])
    green_boundary = float(x[areas["green"]["left"]])
    if orange_boundary >= green_boundary:
        raise ValueError("Граница оранжевой области должна быть меньше зелёной")

    cdf_values = _mixed_cdf_values(k_values, theta, x, sample_counts)
    cdf_orange = float(_mixed_cdf_values(k_values, theta, np.asarray([orange_boundary]), sample_counts)[0])
    cdf_green = float(_mixed_cdf_values(k_values, theta, np.asarray([green_boundary]), sample_counts)[0])
    probabilities = {
        "orange_probability": cdf_orange,
        "blue_probability": cdf_green - cdf_orange,
        "green_probability": 1.0 - cdf_green,
    }
    total = sum(probabilities.values())
    if any(value < -1e-12 or not np.isfinite(value) for value in probabilities.values()):
        raise RuntimeError("Вероятности смешанного распределения некорректны")
    if not np.isclose(total, 1.0, atol=1e-10):
        raise RuntimeError("Сумма вероятностей смешанного распределения не равна 1")

    return {
        "orange_boundary": orange_boundary,
        "green_boundary": green_boundary,
        "cdf_orange": cdf_orange,
        "cdf_green": cdf_green,
        "cdf_values": cdf_values,
        **probabilities,
        "total": float(total),
    }


def calculate_mixed_equal_cdf_areas(k_values, theta, x, sample_counts):
    """Границы метода 2 — квантили 1/3 и 2/3 общей смеси."""
    cdf_values = _mixed_cdf_values(k_values, theta, x, sample_counts)
    # Удаляем возможные численные плато, чтобы интерполяция была однозначной.
    unique_cdf, unique_indices = np.unique(cdf_values, return_index=True)
    if unique_cdf.size < 2 or unique_cdf[-1] < 2.0 / 3.0:
        raise RuntimeError("Диапазон x недостаточен для расчёта квантилей смеси")
    levels = (1.0 / 3.0, 2.0 / 3.0)
    x1, x2 = np.interp(levels, unique_cdf, np.asarray(x)[unique_indices])
    return {
        "orange_boundary": float(x1),
        "green_boundary": float(x2),
        "cdf_orange": float(levels[0]),
        "cdf_green": float(levels[1]),
        "cdf_values": cdf_values,
        "orange_probability": float(1.0 / 3.0),
        "blue_probability": float(1.0 / 3.0),
        "green_probability": float(1.0 / 3.0),
        "total": 1.0,
    }


def calculate_mixed_method3_equal_x_areas(k_values, theta, x, sample_counts):
    """Метод 3: равные интервалы x, вероятности по общей CDF смеси."""
    x_values = np.asarray(x, dtype=float)
    if x_values.ndim != 1 or x_values.size < 2 or np.any(np.diff(x_values) <= 0):
        raise ValueError("x должен быть одномерным, возрастающим и содержать минимум две точки")
    x_min = float(x_values[0])
    x_max = float(x_values[-1])
    width = (x_max - x_min) / 3.0
    boundary_1 = x_min + width
    boundary_2 = x_min + 2.0 * width
    cdf_1 = float(_mixed_cdf_values(k_values, theta, np.asarray([boundary_1]), sample_counts)[0])
    cdf_2 = float(_mixed_cdf_values(k_values, theta, np.asarray([boundary_2]), sample_counts)[0])
    probabilities = {
        "orange_probability": cdf_1,
        "blue_probability": cdf_2 - cdf_1,
        "green_probability": 1.0 - cdf_2,
    }
    if any(not np.isfinite(value) or value < 0 for value in probabilities.values()):
        raise RuntimeError("Теоретические вероятности метода 3 некорректны")
    total = sum(probabilities.values())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise RuntimeError("Теоретические вероятности метода 3 не дают сумму 1")
    return {
        "x_min": x_min,
        "x_max": x_max,
        "orange_boundary": float(boundary_1),
        "green_boundary": float(boundary_2),
        "cdf_orange": cdf_1,
        "cdf_green": cdf_2,
        "cdf_values": _mixed_cdf_values(k_values, theta, x_values, sample_counts),
        **probabilities,
        "total": float(total),
    }


def calculate_mixed_component_probabilities(k_values, theta, left_boundary, right_boundary):
    """Вероятности трёх областей отдельно для каждой Gamma-компоненты."""
    if left_boundary >= right_boundary:
        raise ValueError("Левая граница должна быть меньше правой")
    result = []
    for k in k_values:
        cdf_left = float(gamma.cdf(left_boundary, a=k, scale=theta))
        cdf_right = float(gamma.cdf(right_boundary, a=k, scale=theta))
        result.append({
            "orange": cdf_left,
            "blue": cdf_right - cdf_left,
            "green": 1.0 - cdf_right,
        })
    return result


def calculate_stratified_expected_squared_error(component_probabilities, sample_counts):
    """M[sum_j (p_hat_j-p_j)^2] для фиксированных квот из разных компонент.

    Для области j дисперсия общей доли равна
    sum_i n_i * p_ij * (1-p_ij) / N^2.
    """
    if not component_probabilities or len(component_probabilities) != len(sample_counts):
        raise ValueError("Число наборов вероятностей должно совпадать с числом квот")
    counts = np.asarray(sample_counts, dtype=float)
    if np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("sample_counts должны быть неотрицательными и иметь положительную сумму")
    total_n = float(counts.sum())
    area_names = ("orange", "blue", "green")
    result = 0.0
    for component_index, probabilities in enumerate(component_probabilities):
        values = np.asarray([probabilities[name] for name in area_names], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-10):
            raise ValueError("Вероятности компонента должны быть корректными и давать сумму 1")
        result += counts[component_index] * float(np.sum(values * (1.0 - values))) / (total_n ** 2)
    return float(result)


def run_mixed_replays(k_values, theta, N, replays, x, areas, rng,
                      sample_counts, equal_cdf_areas=None, method3_areas=None):
    """Повторяет эксперименты на смешанных выборках.

    Для каждого replay генерируется исходная выборка размера N для каждого k;
    из каждой без возвращения выбирается квота sample_counts. Одна итоговая
    выборка используется всеми тремя методами в этом replay.
    """
    k_values = list(k_values)
    sample_counts = [int(value) for value in sample_counts]
    if not replays or replays <= 0:
        raise ValueError("replays должно быть > 0")
    if len(k_values) != len(sample_counts) or sum(sample_counts) != int(N):
        raise ValueError("sample_counts должны соответствовать k_values и суммироваться в N")

    results = []
    for replay_number in range(1, int(replays) + 1):
        selected_parts = []
        for k, selected_count in zip(k_values, sample_counts):
            # По условию каждая исходная выборка имеет размер N, даже если
            # в итоговую выборку из неё попадёт только selected_count элементов.
            initial_sample = generate_sample(k=k, theta=theta, N=N, rng=rng)
            if selected_count:
                selected_indices = rng.choice(int(N), size=selected_count, replace=False)
                selected_parts.append(initial_sample[selected_indices])

        sample = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=float)
        rng.shuffle(sample)
        if sample.size != int(N):
            raise RuntimeError(f"Размер смешанной выборки replay {replay_number} не равен N")

        statistics = calculate_replay_statistics(sample=sample, x=x, areas=areas)
        method2_counts = {}
        if equal_cdf_areas is not None:
            b1 = equal_cdf_areas["orange_boundary"]
            b2 = equal_cdf_areas["green_boundary"]
            method2_counts = {
                "method2_orange_count": int(np.sum(sample <= b1)),
                "method2_blue_count": int(np.sum((sample > b1) & (sample < b2))),
                "method2_green_count": int(np.sum(sample >= b2)),
            }
        method3_counts = {}
        if method3_areas is not None:
            b1 = method3_areas["orange_boundary"]
            b2 = method3_areas["green_boundary"]
            method3_counts = {
                "method3_orange_count": int(np.sum(sample <= b1)),
                "method3_blue_count": int(np.sum((sample > b1) & (sample < b2))),
                "method3_green_count": int(np.sum(sample >= b2)),
            }

        all_counts = {
            "orange_count": statistics["orange"]["count"],
            "blue_count": statistics["blue"]["count"],
            "green_count": statistics["green"]["count"],
            **method2_counts,
            **method3_counts,
            "total": statistics["total"],
            "replay": replay_number,
        }
        for k, selected_count in zip(k_values, sample_counts):
            all_counts[f"selected_from_k_{k}"] = int(selected_count)
        results.append(all_counts)

    return results
