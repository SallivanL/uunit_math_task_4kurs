import os
import shutil

import numpy as np

from calc import (
    calculate_densities,
    calculate_areas,
    calculate_cdf_area_probabilities,
    run_replays,
    calculate_replay_totals,
    calculate_area_statistics,
    calculate_replay_ratios,
    calculate_average_replay_ratios,
)

from plotting import (
    plot_three_densities,
    plot_cdf,
    plot_replay_histogram,
    plot_cdf_histogram,
)

from report import save_experiment


def main():
    m = 0.8
    replays = 10

    k_values = [
        1,
        8,
        30,
    ]

    theta_values = [1]
    N_values = [10, 100]

    random_seed = None

    if os.path.exists("results"):
        shutil.rmtree("results")

    rng = np.random.default_rng(random_seed)

    for k in k_values:
        for theta in theta_values:
            for N in N_values:

                # --------------------------------------------------
                # Расчёт плотностей
                # --------------------------------------------------

                x, f, f_min, f_max = calculate_densities(
                    k=k,
                    theta=theta,
                    N=N,
                )

                # --------------------------------------------------
                # Расчёт трёх областей
                # --------------------------------------------------

                areas = calculate_areas(
                    x=x,
                    f=f,
                    f_min=f_min,
                    f_max=f_max,
                    m=m,
                )

                # --------------------------------------------------
                # Основной график распределений
                # --------------------------------------------------

                figure = plot_three_densities(
                    x=x,
                    f=f,
                    f_min=f_min,
                    f_max=f_max,
                    areas=areas,
                    k=k,
                    theta=theta,
                    N=N,
                )

                # --------------------------------------------------
                # Расчёт вероятностей через F(x)
                # --------------------------------------------------

                cdf_area_probabilities = (
                    calculate_cdf_area_probabilities(
                        k=k,
                        theta=theta,
                        x=x,
                        areas=areas,
                    )
                )

                # --------------------------------------------------
                # График F(x) с точками пересечения
                # --------------------------------------------------

                cdf_figure = plot_cdf(
                    x=x,
                    cdf_area_probabilities=cdf_area_probabilities,
                    areas=areas,
                    k=k,
                    theta=theta,
                    N=N,
                )

                # --------------------------------------------------
                # Гистограмма по F(x)
                # --------------------------------------------------

                cdf_histogram = plot_cdf_histogram(
                    cdf_area_probabilities=(
                        cdf_area_probabilities
                    ),
                    k=k,
                    theta=theta,
                    N=N,
                )

                # --------------------------------------------------
                # Проведение replay
                # --------------------------------------------------

                replay_results = run_replays(
                    k=k,
                    theta=theta,
                    N=N,
                    replays=replays,
                    x=x,
                    areas=areas,
                    rng=rng,
                )

                # --------------------------------------------------
                # Итоговые результаты replay
                # --------------------------------------------------

                replay_totals = calculate_replay_totals(
                    replay_results
                )

                # --------------------------------------------------
                # Статистика областей
                # --------------------------------------------------

                area_statistics = calculate_area_statistics(
                    k=k,
                    theta=theta,
                    N=N,
                    x=x,
                    areas=areas,
                    replay_results=replay_results,
                )

                # --------------------------------------------------
                # Доли n / N для каждого replay
                # --------------------------------------------------

                replay_ratios = calculate_replay_ratios(
                    replay_results=replay_results,
                )

                # --------------------------------------------------
                # Средние доли n / N по всем replay
                # --------------------------------------------------

                average_replay_ratios = (
                    calculate_average_replay_ratios(
                        replay_ratios=replay_ratios,
                    )
                )

                # --------------------------------------------------
                # Итоговая гистограмма replay
                # --------------------------------------------------

                histogram = plot_replay_histogram(
                    average_replay_ratios=average_replay_ratios,
                    k=k,
                    theta=theta,
                    N=N,
                    replays=replays,
                )

                # --------------------------------------------------
                # Сохранение результатов
                # --------------------------------------------------

                save_experiment(
                    k=k,
                    theta=theta,
                    N=N,
                    m=m,
                    replays=replays,
                    x=x,
                    areas=areas,
                    figure=figure,
                    cdf_figure=cdf_figure,
                    histogram=histogram,
                    cdf_histogram=cdf_histogram,
                    replay_results=replay_results,
                    replay_ratios=replay_ratios,
                    average_replay_ratios=average_replay_ratios,
                    replay_totals=replay_totals,
                    area_statistics=area_statistics,
                    cdf_area_probabilities=(
                        cdf_area_probabilities
                    ),
                )


if __name__ == "__main__":
    main()
