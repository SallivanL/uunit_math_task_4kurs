"""Математика и проведение экспериментов для четырёх законов распределения.

Поддерживаемые законы:
    1 — равномерный;
    2 — показательный (экспоненциальный);
    3 — нормальный;
    4 — гамма.

Модуль не создаёт графики и не сохраняет файлы.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import uniform, expon, norm, gamma


DISTRIBUTION_NAMES = {
    1: "Равномерное распределение",
    2: "Показательное распределение",
    3: "Нормальное распределение",
    4: "Гамма-распределение",
}


def normalize_distribution_config(distribution: dict[str, Any]) -> dict[str, Any]:
    """Проверяет конфигурацию закона и возвращает нормализованный словарь."""
    if not isinstance(distribution, dict):
        raise TypeError("distribution должен быть словарём конфигурации из main.py")

    try:
        distribution_id = int(distribution["id"])
        key = str(distribution["key"])
        params = dict(distribution["params"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Конфигурация должна содержать поля 'id', 'key' и 'params'."
        ) from exc

    if distribution_id not in DISTRIBUTION_NAMES:
        raise ValueError("Номер закона должен быть от 1 до 4.")

    expected_keys = {1: "uniform", 2: "exponential", 3: "normal", 4: "gamma"}
    if key != expected_keys[distribution_id]:
        raise ValueError(
            f"Для закона №{distribution_id} ожидается ключ '{expected_keys[distribution_id]}'."
        )

    if key == "uniform":
        if "a" not in params or "b" not in params:
            raise ValueError("Для равномерного распределения нужны параметры a и b.")
        a, b = float(params["a"]), float(params["b"])
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            raise ValueError("Для равномерного распределения требуется конечное b > a.")
        params = {"a": a, "b": b}
    elif key == "exponential":
        rate = float(params.get("rate", 0))
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Для показательного распределения rate (λ) должен быть > 0.")
        params = {"rate": rate}
    elif key == "normal":
        mu = float(params.get("mu", 0))
        sigma = float(params.get("sigma", 0))
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Для нормального распределения требуется конечное μ и σ > 0.")
        params = {"mu": mu, "sigma": sigma}
    elif key == "gamma":
        k = float(params.get("k", 0))
        theta = float(params.get("theta", 0))
        if not np.isfinite(k) or not np.isfinite(theta) or k <= 0 or theta <= 0:
            raise ValueError("Для гамма-распределения параметры k и theta должны быть > 0.")
        params = {"k": k, "theta": theta}

    normalized = {
        "id": distribution_id,
        "key": key,
        "label": str(distribution.get("label", DISTRIBUTION_NAMES[distribution_id])),
        "params": params,
    }
    normalized["params_text"] = format_distribution_params(normalized)
    return normalized


def format_distribution_params(distribution: dict[str, Any]) -> str:
    """Формирует человекочитаемую запись параметров закона."""
    key = distribution.get("key")
    params = distribution.get("params", {})
    fmt = lambda value: f"{float(value):g}"
    if key == "uniform":
        return f"a={fmt(params['a'])}, b={fmt(params['b'])}"
    if key == "exponential":
        return f"λ={fmt(params['rate'])}"
    if key == "normal":
        return f"μ={fmt(params['mu'])}, σ={fmt(params['sigma'])}"
    if key == "gamma":
        return f"k={fmt(params['k'])}, θ={fmt(params['theta'])}"
    return "параметры не указаны"


def get_scipy_distribution(distribution: dict[str, Any]):
    """Создаёт замороженный объект scipy.stats для выбранного закона."""
    config = normalize_distribution_config(distribution)
    params = config["params"]
    if config["key"] == "uniform":
        return uniform(loc=params["a"], scale=params["b"] - params["a"])
    if config["key"] == "exponential":
        # В scipy.stats.expon параметр scale равен 1 / rate.
        return expon(scale=1.0 / params["rate"])
    if config["key"] == "normal":
        return norm(loc=params["mu"], scale=params["sigma"])
    if config["key"] == "gamma":
        return gamma(a=params["k"], scale=params["theta"])
    raise ValueError(f"Неподдерживаемый закон: {config['key']}")


def calculate_densities(distribution: dict[str, Any], N: int, points: int = 5000):
    """Рассчитывает исходную плотность и плотности минимума/максимума выборки."""
    if int(N) <= 0:
        raise ValueError("N должно быть > 0")
    if int(points) < 100:
        raise ValueError("Для построения плотностей нужно не менее 100 точек.")

    dist = get_scipy_distribution(distribution)
    # Диапазон расширен с учётом экстремумов выборки размера N.
    # Вероятность пропущенного хвоста исходного закона мала по сравнению с 1/N.
    tail = min(1e-5, 1e-4 / int(N))
    lower = float(dist.ppf(tail))
    upper = float(dist.ppf(1.0 - tail))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Не удалось определить конечный диапазон для выбранного закона.")

    x = np.linspace(lower, upper, int(points))
    f = np.asarray(dist.pdf(x), dtype=float)
    F = np.clip(np.asarray(dist.cdf(x), dtype=float), 0.0, 1.0)
    f_max = int(N) * f * np.power(F, int(N) - 1)
    f_min = int(N) * f * np.power(1.0 - F, int(N) - 1)
    return x, f, f_min, f_max


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Накопленная площадь методом трапеций, включая начальное значение 0."""
    increments = (y[:-1] + y[1:]) * 0.5 * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(increments)))


def find_left_area(x, y, m: float):
    """Находит левую область, содержащую долю m площади под кривой."""
    if not 0 < m <= 1:
        raise ValueError("m должно быть в диапазоне (0, 1].")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(x) != len(y):
        raise ValueError("x и y должны содержать одинаковое число точек (не менее двух).")
    cumulative = _cumulative_trapezoid(y, x)
    total_area = float(cumulative[-1])
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Площадь под кривой должна быть положительной и конечной.")
    right = int(np.searchsorted(cumulative, m * total_area, side="left"))
    right = min(max(right, 1), len(x) - 1)
    area = float(cumulative[right])
    return {
        "left": 0,
        "right": right,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area,
    }


def find_right_area(x, y, m: float):
    """Находит правую область, содержащую долю m площади под кривой."""
    if not 0 < m <= 1:
        raise ValueError("m должно быть в диапазоне (0, 1].")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(x) != len(y):
        raise ValueError("x и y должны содержать одинаковое число точек (не менее двух).")
    cumulative_from_left = _cumulative_trapezoid(y, x)
    total_area = float(cumulative_from_left[-1])
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Площадь под кривой должна быть положительной и конечной.")
    target_from_left = total_area * (1.0 - m)
    left = int(np.searchsorted(cumulative_from_left, target_from_left, side="left"))
    left = min(max(left, 0), len(x) - 2)
    area = float(total_area - cumulative_from_left[left])
    return {
        "left": left,
        "right": len(x) - 1,
        "area": area,
        "total_area": total_area,
        "ratio": area / total_area,
    }


def find_middle_area(x, left_index: int, right_index: int):
    """Формирует центральную область между двумя границами."""
    if left_index >= right_index:
        raise ValueError(
            "Границы областей пересеклись. Измените параметры распределения, N или m."
        )
    return {
        "left": int(left_index),
        "right": int(right_index),
        "left_boundary": float(x[left_index]),
        "right_boundary": float(x[right_index]),
    }


def calculate_areas(x, f, f_min, f_max, m: float):
    """Формирует три области: левую по f_min, центральную и правую по f_max."""
    orange = find_left_area(x=x, y=f_min, m=m)
    green = find_right_area(x=x, y=f_max, m=m)
    blue = find_middle_area(x=x, left_index=orange["right"], right_index=green["left"])
    return {"orange": orange, "blue": blue, "green": green}


def calculate_cdf_area_probabilities(distribution, x, areas):
    """Рассчитывает теоретические вероятности областей через CDF выбранного закона."""
    dist = get_scipy_distribution(distribution)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x не должен быть пустым.")
    x1 = float(x[areas["orange"]["right"]])
    x2 = float(x[areas["green"]["left"]])
    if x1 >= x2:
        raise ValueError("Граница оранжевой области должна быть меньше границы зелёной области.")
    cdf_x1 = float(dist.cdf(x1))
    cdf_x2 = float(dist.cdf(x2))
    p1 = cdf_x1
    p2 = cdf_x2 - cdf_x1
    p3 = 1.0 - cdf_x2
    total = p1 + p2 + p3
    if not np.isclose(total, 1.0, atol=1e-12):
        raise RuntimeError("Теоретические вероятности через CDF не дают сумму 1.")
    return {
        "orange_boundary": x1,
        "green_boundary": x2,
        "cdf_orange": cdf_x1,
        "cdf_green": cdf_x2,
        "cdf_values": np.asarray(dist.cdf(x), dtype=float),
        "orange_probability": float(p1),
        "blue_probability": float(p2),
        "green_probability": float(p3),
        "total": float(total),
    }


def calculate_equal_cdf_areas(distribution, x=None):
    """Границы метода 2: деление вероятностной массы на три равные части."""
    dist = get_scipy_distribution(distribution)
    levels = (1.0 / 3.0, 2.0 / 3.0)
    x1, x2 = dist.ppf(levels)
    result = {
        "orange_boundary": float(x1),
        "green_boundary": float(x2),
        "cdf_orange": float(levels[0]),
        "cdf_green": float(levels[1]),
        "orange_probability": float(1.0 / 3.0),
        "blue_probability": float(1.0 / 3.0),
        "green_probability": float(1.0 / 3.0),
        "total": 1.0,
    }
    if x is not None:
        result["cdf_values"] = np.asarray(dist.cdf(x), dtype=float)
    return result


def generate_sample(distribution, N: int, rng: np.random.Generator):
    """Генерирует одну выборку из N значений выбранного закона."""
    if int(N) <= 0:
        raise ValueError("N должно быть > 0.")
    if rng is None:
        raise ValueError("Для генерации необходимо передать генератор rng.")
    dist = get_scipy_distribution(distribution)
    return np.asarray(dist.rvs(size=int(N), random_state=rng), dtype=float)


def calculate_replay_statistics(sample, x, areas):
    """Классифицирует каждое значение выборки ровно в одну из трёх областей."""
    sample = np.asarray(sample, dtype=float)
    x = np.asarray(x, dtype=float)
    orange_boundary = float(x[areas["orange"]["right"]])
    green_boundary = float(x[areas["green"]["left"]])
    orange_mask = sample <= orange_boundary
    blue_mask = (sample > orange_boundary) & (sample < green_boundary)
    green_mask = sample >= green_boundary
    orange_count = int(np.sum(orange_mask))
    blue_count = int(np.sum(blue_mask))
    green_count = int(np.sum(green_mask))
    total_count = int(sample.size)
    if orange_count + blue_count + green_count != total_count:
        raise RuntimeError("Не удалось классифицировать все элементы выборки.")
    return {
        "orange": {"count": orange_count, "ratio": orange_count / total_count},
        "blue": {"count": blue_count, "ratio": blue_count / total_count},
        "green": {"count": green_count, "ratio": green_count / total_count},
        "total": total_count,
        "boundaries": {"orange": orange_boundary, "green": green_boundary},
    }


def run_replays(distribution, N, replays, x, areas, rng, equal_cdf_areas=None):
    """Выполняет replay; оба метода используют одну и ту же выборку в каждом replay."""
    if int(replays) <= 0:
        raise ValueError("replays должно быть > 0.")
    results = []
    b1 = b2 = None
    if equal_cdf_areas is not None:
        b1 = float(equal_cdf_areas["orange_boundary"])
        b2 = float(equal_cdf_areas["green_boundary"])
    for replay_number in range(1, int(replays) + 1):
        sample = generate_sample(distribution=distribution, N=N, rng=rng)
        statistics = calculate_replay_statistics(sample=sample, x=x, areas=areas)
        method2_counts = {}
        if equal_cdf_areas is not None:
            method2_counts = {
                "method2_orange_count": int(np.sum(sample <= b1)),
                "method2_blue_count": int(np.sum((sample > b1) & (sample < b2))),
                "method2_green_count": int(np.sum(sample >= b2)),
            }
            if sum(method2_counts.values()) != len(sample):
                raise RuntimeError("Метод 2 не классифицировал все элементы выборки.")
        results.append({
            "replay": replay_number,
            "orange_count": statistics["orange"]["count"],
            "blue_count": statistics["blue"]["count"],
            "green_count": statistics["green"]["count"],
            **method2_counts,
            "total": statistics["total"],
        })
    return results


def calculate_replay_ratios(replay_results):
    """Рассчитывает доли n/N по каждой области для каждого replay."""
    if not replay_results:
        raise ValueError("replay_results не должен быть пустым.")
    ratios = []
    for result in replay_results:
        total = int(result["total"])
        if total <= 0:
            raise ValueError("Количество элементов replay должно быть > 0.")
        values = {
            "orange_ratio": result["orange_count"] / total,
            "blue_ratio": result["blue_count"] / total,
            "green_ratio": result["green_count"] / total,
        }
        if not np.isclose(sum(values.values()), 1.0, atol=1e-12):
            raise RuntimeError("Сумма долей областей для replay не равна 1.")
        ratios.append({"replay": result["replay"], **{k: float(v) for k, v in values.items()}})
    return ratios


def calculate_average_replay_ratios(replay_ratios):
    """Рассчитывает средние доли областей по всем replay."""
    if not replay_ratios:
        raise ValueError("replay_ratios не должен быть пустым.")
    values = {
        "orange_ratio": float(np.mean([r["orange_ratio"] for r in replay_ratios])),
        "blue_ratio": float(np.mean([r["blue_ratio"] for r in replay_ratios])),
        "green_ratio": float(np.mean([r["green_ratio"] for r in replay_ratios])),
    }
    total = sum(values.values())
    if not np.isclose(total, 1.0, atol=1e-12):
        raise RuntimeError("Средние доли трёх областей не дают сумму 1.")
    return {**values, "total": float(total)}


def calculate_replay_totals(replay_results):
    """Суммирует результаты основного метода по всем replay."""
    if not replay_results:
        raise ValueError("replay_results не должен быть пустым.")
    total = sum(int(r["total"]) for r in replay_results)
    counts = {area: sum(int(r[f"{area}_count"]) for r in replay_results)
              for area in ("orange", "blue", "green")}
    if sum(counts.values()) != total:
        raise RuntimeError("Сумма трёх областей не равна общему количеству элементов.")
    result = {"total": total}
    for area, count in counts.items():
        result[f"{area}_count"] = count
        result[f"{area}_ratio"] = count / total
    return result


def calculate_method2_replay_totals(replay_results):
    """Суммирует фактические количества точек для метода 2."""
    if not replay_results or any("method2_orange_count" not in r for r in replay_results):
        raise ValueError("В replay_results отсутствуют результаты метода 2.")
    total = sum(int(r["total"]) for r in replay_results)
    counts = {
        "orange": sum(int(r["method2_orange_count"]) for r in replay_results),
        "blue": sum(int(r["method2_blue_count"]) for r in replay_results),
        "green": sum(int(r["method2_green_count"]) for r in replay_results),
    }
    if sum(counts.values()) != total:
        raise RuntimeError("Сумма областей метода 2 не равна общему количеству элементов.")
    result = {"total": total}
    for area, count in counts.items():
        result[f"{area}_count"] = count
        result[f"{area}_ratio"] = count / total
    return result


def calculate_area_statistics(distribution, N, x, areas, replay_results):
    """Теоретическая и экспериментальная статистика по трём областям."""
    dist = get_scipy_distribution(distribution)
    x = np.asarray(x, dtype=float)
    orange_boundary = float(x[areas["orange"]["right"]])
    green_boundary = float(x[areas["green"]["left"]])
    cdf_orange = float(dist.cdf(orange_boundary))
    cdf_green = float(dist.cdf(green_boundary))
    probabilities = {
        "orange": cdf_orange,
        "blue": cdf_green - cdf_orange,
        "green": 1.0 - cdf_green,
    }
    if not np.isclose(sum(probabilities.values()), 1.0, atol=1e-10):
        raise RuntimeError("Теоретические вероятности трёх областей не дают сумму 1.")

    statistics = {}
    total_replay_elements = len(replay_results) * int(N)
    for area in ("orange", "blue", "green"):
        counts = np.asarray([r[f"{area}_count"] for r in replay_results], dtype=float)
        probability = probabilities[area]
        variance = float(np.var(counts, ddof=0))
        statistics[area] = {
            "mathematical_expectation": float(N * probability),
            "frequency": float(np.sum(counts) / total_replay_elements),
            "mean": float(np.mean(counts)),
            "standard_deviation": float(np.sqrt(variance)),
            "variance": variance,
            "theoretical_probability": float(probability),
        }
    return statistics
