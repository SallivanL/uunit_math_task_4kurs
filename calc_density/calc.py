# calc.py

import numpy as np
from scipy.stats import gamma


def calculate_densities(k, theta, N, points=5000):
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
        scale=theta
    )

    x = np.linspace(
        0,
        x_max,
        points
    )

    f = gamma.pdf(
        x,
        a=k,
        scale=theta
    )

    F = gamma.cdf(
        x,
        a=k,
        scale=theta
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


def find_peak_area(x, y, m):
    peak_index = np.argmax(y)

    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    left = peak_index
    right = peak_index

    while left > 0 or right < len(x) - 1:
        left_area = np.trapezoid(
            y[left:peak_index + 1],
            x[left:peak_index + 1]
        )

        right_area = np.trapezoid(
            y[peak_index:right + 1],
            x[peak_index:right + 1]
        )

        if left_area + right_area >= target_area:
            break

        left_candidate = (
            y[left - 1]
            if left > 0
            else -1
        )

        right_candidate = (
            y[right + 1]
            if right < len(x) - 1
            else -1
        )

        if left_candidate >= right_candidate and left > 0:
            left -= 1
        elif right < len(x) - 1:
            right += 1
        else:
            break

    area = np.trapezoid(
        y[left:right + 1],
        x[left:right + 1]
    )

    return {
        "left": left,
        "right": right,
        "peak": peak_index,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area
    }


def find_left_area(x, y, m):
    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    right = 0

    for i in range(1, len(x)):
        area = np.trapezoid(
            y[:i + 1],
            x[:i + 1]
        )

        if area >= target_area:
            right = i
            break

    area = np.trapezoid(
        y[:right + 1],
        x[:right + 1]
    )

    return {
        "left": 0,
        "right": right,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area
    }


def find_right_area(x, y, m):
    total_area = np.trapezoid(y, x)
    target_area = m * total_area

    left = len(x) - 1

    for i in range(len(x) - 2, -1, -1):
        area = np.trapezoid(
            y[i:],
            x[i:]
        )

        if area >= target_area:
            left = i
            break

    area = np.trapezoid(
        y[left:],
        x[left:]
    )

    return {
        "left": left,
        "right": len(x) - 1,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area
    }


def calculate_areas(x, f, f_min, f_max, m):
    return {
        "f": find_left_area(x, f, m),
        "f_min": find_left_area(x, f_min, m),
        "f_max": find_right_area(x, f_max, m),
    }


def generate_sample(k, theta, N, rng):
    """
    Генерирует одну фактическую выборку из N элементов
    исходного распределения Gamma(k, theta).
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
        size=int(N)
    )


def calculate_replay_statistics(sample, x, areas):
    """
    Классифицирует каждый фактически сгенерированный элемент
    по трём закрашенным областям.

    Оранжевая:
        x <= граница f_min

    Синяя:
        граница f_min < x <= граница f

    Зелёная:
        x >= граница f_max

    Остальные элементы не попадают ни в одну закрашенную область.
    """

    orange_boundary = x[areas["f_min"]["right"]]
    blue_boundary = x[areas["f"]["right"]]
    green_boundary = x[areas["f_max"]["left"]]

    orange_mask = sample <= orange_boundary

    blue_mask = (
        (sample > orange_boundary)
        & (sample <= blue_boundary)
    )

    green_mask = sample >= green_boundary

    orange_count = int(np.sum(orange_mask))
    blue_count = int(np.sum(blue_mask))
    green_count = int(np.sum(green_mask))

    total_count = len(sample)

    covered_count = (
        orange_count
        + blue_count
        + green_count
    )

    outside_count = total_count - covered_count

    return {
        "orange": {
            "count": orange_count,
            "ratio": orange_count / total_count
        },
        "blue": {
            "count": blue_count,
            "ratio": blue_count / total_count
        },
        "green": {
            "count": green_count,
            "ratio": green_count / total_count
        },
        "outside": {
            "count": outside_count,
            "ratio": outside_count / total_count
        },
        "total": total_count,
        "boundaries": {
            "orange": orange_boundary,
            "blue": blue_boundary,
            "green": green_boundary
        }
    }


def run_replays(k, theta, N, replays, x, areas, rng):
    """
    Выполняет replays независимых экспериментов.

    Каждый replay:
        1. генерирует N реальных элементов;
        2. определяет, куда попал каждый элемент;
        3. сохраняет фактическое количество элементов.
    """

    if replays <= 0:
        raise ValueError("replays должно быть > 0")

    results = []

    for replay in range(1, replays + 1):
        sample = generate_sample(
            k=k,
            theta=theta,
            N=N,
            rng=rng
        )

        statistics = calculate_replay_statistics(
            sample=sample,
            x=x,
            areas=areas
        )

        results.append({
            "replay": replay,
            "orange_count": statistics["orange"]["count"],
            "blue_count": statistics["blue"]["count"],
            "green_count": statistics["green"]["count"],
            "outside_count": statistics["outside"]["count"],
            "total": statistics["total"]
        })

    return results


def calculate_replay_totals(replay_results):
    """
    Суммирует фактические результаты всех replay.
    """

    total = sum(result["total"] for result in replay_results)

    orange_count = sum(result["orange_count"] for result in replay_results)
    blue_count = sum(result["blue_count"] for result in replay_results)
    green_count = sum(result["green_count"] for result in replay_results)
    outside_count = sum(result["outside_count"] for result in replay_results)

    return {
        "orange_count": orange_count,
        "orange_ratio": orange_count / total,
        "blue_count": blue_count,
        "blue_ratio": blue_count / total,
        "green_count": green_count,
        "green_ratio": green_count / total,
        "outside_count": outside_count,
        "outside_ratio": outside_count / total,
        "total": total
    }