import numpy as np

from calc import (
    calculate_densities,
    calculate_areas,
    calculate_cdf_area_probabilities,
    calculate_equal_cdf_areas,
    calculate_method2_replay_totals,
    calculate_chi_squared_comparison,
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
    plot_replays_histogram,
    plot_method2_replays_histogram,
    plot_cdf_histogram,
    plot_equal_cdf,
    plot_equal_cdf_histogram,
)

from report import clear_results, save_experiment


def main():
    m = 0.8
    replays = 10

    k_values = [
        1,
        8,
        30,
    ]

    theta_values = [1]
    N_values = [10, 100, 1000]

    random_seed = None

    clear_results()

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
                # Метод 2: равное деление диапазона F(x) на три части
                # --------------------------------------------------

                equal_cdf_areas = calculate_equal_cdf_areas(
                    k=k, theta=theta, x=x,
                )
                equal_cdf_figure = plot_equal_cdf(
                    x=x, equal_cdf_areas=equal_cdf_areas,
                    k=k, theta=theta, N=N,
                )
                equal_cdf_histogram = plot_equal_cdf_histogram(
                    equal_cdf_areas=equal_cdf_areas,
                    k=k, theta=theta, N=N,
                )

                # --------------------------------------------------
                # Проведение replay: оба метода используют одни выборки
                # --------------------------------------------------

                replay_results = run_replays(
                    k=k,
                    theta=theta,
                    N=N,
                    replays=replays,
                    x=x,
                    areas=areas,
                    rng=rng,
                    equal_cdf_areas=equal_cdf_areas,
                )

                # --------------------------------------------------
                # Итоговые результаты replay
                # --------------------------------------------------

                replay_totals = calculate_replay_totals(
                    replay_results
                )
                method2_replay_totals = calculate_method2_replay_totals(
                    replay_results
                )

                # --------------------------------------------------
                # Сравнение теоретических и экспериментальных
                # вероятностей по формуле chi² для обоих методов
                # --------------------------------------------------

                method1_chi_squared = calculate_chi_squared_comparison(
                    theoretical_probabilities={
                        "orange": cdf_area_probabilities["orange_probability"],
                        "blue": cdf_area_probabilities["blue_probability"],
                        "green": cdf_area_probabilities["green_probability"],
                    },
                    observed_counts={
                        "orange": replay_totals["orange_count"],
                        "blue": replay_totals["blue_count"],
                        "green": replay_totals["green_count"],
                    },
                )

                method2_chi_squared = calculate_chi_squared_comparison(
                    theoretical_probabilities={
                        "orange": equal_cdf_areas["orange_probability"],
                        "blue": equal_cdf_areas["blue_probability"],
                        "green": equal_cdf_areas["green_probability"],
                    },
                    observed_counts={
                        "orange": method2_replay_totals["orange_count"],
                        "blue": method2_replay_totals["blue_count"],
                        "green": method2_replay_totals["green_count"],
                    },
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
                # Один рисунок со всеми replay-гистограммами
                # --------------------------------------------------

                replay_histogram = plot_replays_histogram(
                    replay_results=replay_results,
                    N=N,
                    k=k,
                    theta=theta,
                )

                # Единый рисунок со всеми replay-гистограммами метода 2
                method2_replay_histogram = plot_method2_replays_histogram(
                    replay_results=replay_results,
                    N=N,
                    k=k,
                    theta=theta,
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
                    replay_histogram=replay_histogram,
                    method2_replay_histogram=method2_replay_histogram,
                    cdf_histogram=cdf_histogram,
                    equal_cdf_figure=equal_cdf_figure,
                    equal_cdf_histogram=equal_cdf_histogram,
                    equal_cdf_areas=equal_cdf_areas,
                    method2_replay_totals=method2_replay_totals,
                    method1_chi_squared=method1_chi_squared,
                    method2_chi_squared=method2_chi_squared,
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
