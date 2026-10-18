"""Точка входа: orchestration экспериментов по выбранным законам распределения."""

import numpy as np

from calc import (
    calculate_densities,
    calculate_areas,
    calculate_cdf_area_probabilities,
    calculate_equal_cdf_areas,
    run_replays,
    calculate_method2_replay_totals,
    calculate_replay_totals,
    calculate_area_statistics,
    calculate_replay_ratios,
    calculate_average_replay_ratios,
)
from plotting import (
    plot_three_densities,
    plot_cdf,
    plot_replay_histogram,
    plot_replays_histogram,
    plot_method2_replays_histogram,
    plot_cdf_histogram,
    plot_equal_cdf,
    plot_equal_cdf_histogram,
)
from report import clear_results, save_experiment


# Номера законов, которые должны быть запущены:
# 1 — равномерный; 2 — показательный; 3 — нормальный; 4 — гамма.
distribution_ids = [1, 2, 3, 4]

# Список конфигураций. Несколько конфигураций одного закона разрешены.
distribution_configs = [
    {"id": 1, "key": "uniform", "label": "Равномерное распределение", "params": {"a": 0, "b": 1}},
    {"id": 2, "key": "exponential", "label": "Показательное распределение", "params": {"rate": 1}},
    {"id": 3, "key": "normal", "label": "Нормальное распределение", "params": {"mu": 0, "sigma": 1}},
    {"id": 4, "key": "gamma", "label": "Гамма-распределение", "params": {"k": 1, "theta": 1}},
    {"id": 4, "key": "gamma", "label": "Гамма-распределение", "params": {"k": 8, "theta": 1}},
    {"id": 4, "key": "gamma", "label": "Гамма-распределение", "params": {"k": 30, "theta": 1}},
]

# Общие параметры эксперимента.
m = 0.8
replays = 10
N_values = [10, 100]
random_seed = None  # Укажи целое число (например, 42) для повторяемых результатов.


def main():
    selected_configs = [config for config in distribution_configs if config["id"] in distribution_ids]
    if not selected_configs:
        raise ValueError("В distribution_ids необходимо указать хотя бы один закон от 1 до 4.")
    invalid_ids = set(distribution_ids) - {1, 2, 3, 4}
    if invalid_ids:
        raise ValueError(f"Неизвестные номера законов: {sorted(invalid_ids)}")

    # Очищаем предыдущие результаты один раз перед всей серией запусков.
    clear_results()
    rng = np.random.default_rng(random_seed)

    for distribution in selected_configs:
        for N in N_values:
            x, f, f_min, f_max = calculate_densities(
                distribution=distribution,
                N=N,
            )

            areas = calculate_areas(
                x=x,
                f=f,
                f_min=f_min,
                f_max=f_max,
                m=m,
            )

            cdf_area_probabilities = calculate_cdf_area_probabilities(
                distribution=distribution,
                x=x,
                areas=areas,
            )

            equal_cdf_areas = calculate_equal_cdf_areas(
                distribution=distribution,
                x=x,
            )

            replay_results = run_replays(
                distribution=distribution,
                N=N,
                replays=replays,
                x=x,
                areas=areas,
                rng=rng,
                equal_cdf_areas=equal_cdf_areas,
            )

            replay_totals = calculate_replay_totals(replay_results)
            method2_replay_totals = calculate_method2_replay_totals(replay_results)
            area_statistics = calculate_area_statistics(
                distribution=distribution,
                N=N,
                x=x,
                areas=areas,
                replay_results=replay_results,
            )
            replay_ratios = calculate_replay_ratios(replay_results)
            average_replay_ratios = calculate_average_replay_ratios(replay_ratios)

            # Построение графиков. plotting.py не сохраняет файлы.
            figure = plot_three_densities(
                x=x, f=f, f_min=f_min, f_max=f_max, areas=areas,
                distribution=distribution, N=N,
            )
            cdf_figure = plot_cdf(
                x=x, cdf_area_probabilities=cdf_area_probabilities,
                distribution=distribution, N=N,
            )
            histogram = plot_replay_histogram(
                average_replay_ratios=average_replay_ratios,
                distribution=distribution, N=N, replays=replays,
            )
            replay_histogram = plot_replays_histogram(
                replay_results=replay_results, N=N, distribution=distribution,
            )
            method2_replay_histogram = plot_method2_replays_histogram(
                replay_results=replay_results, N=N, distribution=distribution,
            )
            cdf_histogram = plot_cdf_histogram(
                cdf_area_probabilities=cdf_area_probabilities,
                distribution=distribution, N=N,
            )
            equal_cdf_figure = plot_equal_cdf(
                x=x, equal_cdf_areas=equal_cdf_areas,
                distribution=distribution, N=N,
            )
            equal_cdf_histogram = plot_equal_cdf_histogram(
                equal_cdf_areas=equal_cdf_areas,
                distribution=distribution, N=N,
            )

            # report.py — единственный модуль, который сохраняет результаты на диск.
            output_dir = save_experiment(
                distribution=distribution,
                N=N,
                m=m,
                replays=replays,
                x=x,
                areas=areas,
                figure=figure,
                cdf_figure=cdf_figure,
                histogram=histogram,
                replay_histogram=replay_histogram,
                method2_replay_histogram=method2_replay_histogram,
                cdf_histogram=cdf_histogram,
                equal_cdf_figure=equal_cdf_figure,
                equal_cdf_histogram=equal_cdf_histogram,
                equal_cdf_areas=equal_cdf_areas,
                method2_replay_totals=method2_replay_totals,
                replay_results=replay_results,
                replay_ratios=replay_ratios,
                average_replay_ratios=average_replay_ratios,
                replay_totals=replay_totals,
                area_statistics=area_statistics,
                cdf_area_probabilities=cdf_area_probabilities,
            )
            print(f"Готово: {distribution['label']}, {distribution['params']}, N={N} -> {output_dir}")


if __name__ == "__main__":
    main()
