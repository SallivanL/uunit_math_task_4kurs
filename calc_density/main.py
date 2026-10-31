import numpy as np

from calc import (
    calculate_densities,
    calculate_areas,
    calculate_cdf_area_probabilities,
    calculate_equal_cdf_areas,
    calculate_method3_equal_x_areas,
    calculate_method2_replay_totals,
    calculate_method3_replay_totals,
    calculate_chi_squared_comparison,
    calculate_expected_squared_error,
    run_replays,
    calculate_replay_totals,
    calculate_area_statistics,
    calculate_replay_ratios,
    calculate_average_replay_ratios,
    calculate_mixed_sample_counts,
    calculate_mixed_densities,
    calculate_mixed_cdf_area_probabilities,
    calculate_mixed_equal_cdf_areas,
    calculate_mixed_method3_equal_x_areas,
    calculate_mixed_component_probabilities,
    calculate_stratified_expected_squared_error,
    run_mixed_replays,
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
    plot_method3_distribution,
    plot_method3_cdf,
    plot_method3_theoretical_histogram,
    plot_method3_replays_histogram,
    plot_comparison_three_methods,
)

from report import clear_results, save_experiment, save_comparison_graph


def normalize_experiment_groups(massive_k, k_values):
    """Преобразует massive_k в список независимых групп k.

    Поддерживаемые варианты:
        [1, 8, 15]                         -> одна смешанная группа
        [[1, 8, 15], [2, 4, 6]]           -> две независимые группы
        None                              -> прежний режим по k_values
    """
    if massive_k is None:
        raw_groups = [[value] for value in k_values]
    else:
        if not isinstance(massive_k, (list, tuple)) or not massive_k:
            raise ValueError("massive_k должен быть непустым списком или None")

        def is_k_value(value):
            return (
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
            )

        if all(is_k_value(value) for value in massive_k):
            raw_groups = [list(massive_k)]
        else:
            raw_groups = []
            for group_index, group in enumerate(massive_k, start=1):
                if not isinstance(group, (list, tuple)) or not group:
                    raise ValueError(
                        f"Элемент massive_k с номером {group_index} должен быть "
                        "непустым списком значений k"
                    )
                if not all(is_k_value(value) for value in group):
                    raise ValueError(
                        f"Группа massive_k[{group_index - 1}] должна содержать "
                        "только числовые значения k; вложенность глубже двух уровней не поддерживается"
                    )
                raw_groups.append(list(group))

    normalized_groups = []
    folder_names = set()
    for group_index, group in enumerate(raw_groups, start=1):
        normalized = [value.item() if isinstance(value, np.generic) else value for value in group]
        for value in normalized:
            try:
                valid = np.isfinite(float(value)) and float(value) > 0
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(
                    f"Все значения k должны быть конечными числами больше нуля; "
                    f"ошибка в группе {group_index}: {value!r}"
                )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"В группе {group_index} значения k не должны повторяться")

        folder_name = "_".join(str(value) for value in normalized)
        if folder_name in folder_names:
            raise ValueError(
                f"Группы massive_k приводят к одинаковой папке k_{folder_name}; "
                "названия групп должны быть уникальными"
            )
        folder_names.add(folder_name)
        normalized_groups.append(normalized)

    return normalized_groups


def main():
    m = 0.8
    replays = 10

    # Каждый внутренний список — отдельный независимый смешанный эксперимент.
    # Допустим и плоский вариант massive_k = [1, 8, 15] для одной группы.
    # Для прежнего режима отдельных экспериментов задай massive_k = None.
    massive_k = [[1, 8, 15], [2, 4, 6], [5, 15, 35]]
    k_values = [1, 8]  # Используется только когда massive_k = None.

    theta_values = [1]
    N_values = [20, 100, 500, 1000]

    random_seed = None

    clear_results()

    rng = np.random.default_rng(random_seed)

    experiment_groups = normalize_experiment_groups(massive_k, k_values)

    for k_group in experiment_groups:
        k = int(k_group[0]) if len(k_group) == 1 else list(k_group)
        k_folder = "_".join(str(value) for value in k_group)
        is_mixed = len(k_group) > 1
        for theta in theta_values:
            # Аналитические значения M[X²] накапливаются отдельно для каждой пары (k, theta).
            comparison_results = {
                "N": [],
                "method_1": [],
                "method_2": [],
                "method_3": [],
            }

            for N in N_values:
                # --------------------------------------------------
                # Расчёт плотностей
                # --------------------------------------------------

                if is_mixed:
                    sample_counts = calculate_mixed_sample_counts(k_group, N)
                    x, f, f_min, f_max = calculate_mixed_densities(
                        k_values=k_group,
                        theta=theta,
                        N=N,
                        sample_counts=sample_counts,
                    )
                else:
                    sample_counts = [N]
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

                if is_mixed:
                    cdf_area_probabilities = calculate_mixed_cdf_area_probabilities(
                        k_values=k_group,
                        theta=theta,
                        x=x,
                        areas=areas,
                        sample_counts=sample_counts,
                    )
                else:
                    cdf_area_probabilities = calculate_cdf_area_probabilities(
                        k=k,
                        theta=theta,
                        x=x,
                        areas=areas,
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

                if is_mixed:
                    equal_cdf_areas = calculate_mixed_equal_cdf_areas(
                        k_values=k_group,
                        theta=theta,
                        x=x,
                        sample_counts=sample_counts,
                    )
                else:
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
                # Метод 3: деление диапазона x на три равных интервала.
                # Для изображений используется префикс method_3.
                # --------------------------------------------------

                if is_mixed:
                    method3_areas = calculate_mixed_method3_equal_x_areas(
                        k_values=k_group,
                        theta=theta,
                        x=x,
                        sample_counts=sample_counts,
                    )
                else:
                    method3_areas = calculate_method3_equal_x_areas(
                        k=k, theta=theta, x=x,
                    )
                method3_distribution_figure = plot_method3_distribution(
                    x=x, f=f, f_min=f_min, f_max=f_max, method3_areas=method3_areas,
                    k=k, theta=theta, N=N,
                )
                method3_cdf_figure = plot_method3_cdf(
                    x=x, method3_areas=method3_areas,
                    k=k, theta=theta, N=N,
                )
                method3_theoretical_histogram = plot_method3_theoretical_histogram(
                    method3_areas=method3_areas,
                    k=k, theta=theta, N=N,
                )

                # --------------------------------------------------
                # Проведение replay: все методы используют одни выборки
                # --------------------------------------------------

                if is_mixed:
                    replay_results = run_mixed_replays(
                        k_values=k_group,
                        theta=theta,
                        N=N,
                        replays=replays,
                        x=x,
                        areas=areas,
                        rng=rng,
                        sample_counts=sample_counts,
                        equal_cdf_areas=equal_cdf_areas,
                        method3_areas=method3_areas,
                    )
                else:
                    replay_results = run_replays(
                        k=k,
                        theta=theta,
                        N=N,
                        replays=replays,
                        x=x,
                        areas=areas,
                        rng=rng,
                        equal_cdf_areas=equal_cdf_areas,
                        method3_areas=method3_areas,
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
                method3_replay_totals = calculate_method3_replay_totals(
                    replay_results
                )

                # --------------------------------------------------
                # Сравнение теоретических и экспериментальных
                # вероятностей по формуле chi² для трёх методов
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

                method3_chi_squared = calculate_chi_squared_comparison(
                    theoretical_probabilities={
                        "orange": method3_areas["orange_probability"],
                        "blue": method3_areas["blue_probability"],
                        "green": method3_areas["green_probability"],
                    },
                    observed_counts={
                        "orange": method3_replay_totals["orange_count"],
                        "blue": method3_replay_totals["blue_count"],
                        "green": method3_replay_totals["green_count"],
                    },
                )

                # --------------------------------------------------
                # Аналитическое математическое ожидание квадрата ошибки
                # M[X²] = sum_j p_j * (1 - p_j) / N для каждого метода.
                # --------------------------------------------------

                method1_probabilities = {
                    "orange": cdf_area_probabilities["orange_probability"],
                    "blue": cdf_area_probabilities["blue_probability"],
                    "green": cdf_area_probabilities["green_probability"],
                }
                method2_probabilities = {
                    "orange": equal_cdf_areas["orange_probability"],
                    "blue": equal_cdf_areas["blue_probability"],
                    "green": equal_cdf_areas["green_probability"],
                }
                method3_probabilities = {
                    "orange": method3_areas["orange_probability"],
                    "blue": method3_areas["blue_probability"],
                    "green": method3_areas["green_probability"],
                }

                if is_mixed:
                    method1_component_probabilities = calculate_mixed_component_probabilities(
                        k_values=k_group,
                        theta=theta,
                        left_boundary=cdf_area_probabilities["orange_boundary"],
                        right_boundary=cdf_area_probabilities["green_boundary"],
                    )
                    method2_component_probabilities = calculate_mixed_component_probabilities(
                        k_values=k_group,
                        theta=theta,
                        left_boundary=equal_cdf_areas["orange_boundary"],
                        right_boundary=equal_cdf_areas["green_boundary"],
                    )
                    method3_component_probabilities = calculate_mixed_component_probabilities(
                        k_values=k_group,
                        theta=theta,
                        left_boundary=method3_areas["orange_boundary"],
                        right_boundary=method3_areas["green_boundary"],
                    )
                    method1_expected_squared_error = calculate_stratified_expected_squared_error(
                        method1_component_probabilities, sample_counts
                    )
                    method2_expected_squared_error = calculate_stratified_expected_squared_error(
                        method2_component_probabilities, sample_counts
                    )
                    method3_expected_squared_error = calculate_stratified_expected_squared_error(
                        method3_component_probabilities, sample_counts
                    )
                else:
                    method1_expected_squared_error = calculate_expected_squared_error(
                        theoretical_probabilities=method1_probabilities,
                        sample_size=N,
                    )
                    method2_expected_squared_error = calculate_expected_squared_error(
                        theoretical_probabilities=method2_probabilities,
                        sample_size=N,
                    )
                    method3_expected_squared_error = calculate_expected_squared_error(
                        theoretical_probabilities=method3_probabilities,
                        sample_size=N,
                    )

                comparison_results["N"].append(int(N))
                comparison_results["method_1"].append(
                    method1_expected_squared_error
                )
                comparison_results["method_2"].append(
                    method2_expected_squared_error
                )
                comparison_results["method_3"].append(
                    method3_expected_squared_error
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
                    theoretical_probabilities=method1_probabilities if is_mixed else None,
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

                method3_replay_histogram = plot_method3_replays_histogram(
                    replay_results=replay_results,
                    N=N,
                    k=k,
                    theta=theta,
                )

                # --------------------------------------------------
                # Передача рассчитанных результатов и графиков в report.py
                # --------------------------------------------------

                save_experiment(
                    k=k_folder,
                    k_display=k,
                    sample_counts=sample_counts if is_mixed else None,
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
                    method3_distribution_figure=method3_distribution_figure,
                    method3_cdf_figure=method3_cdf_figure,
                    method3_theoretical_histogram=method3_theoretical_histogram,
                    method3_replay_histogram=method3_replay_histogram,
                    method3_areas=method3_areas,
                    method3_replay_totals=method3_replay_totals,
                    method3_chi_squared=method3_chi_squared,
                )

            # После обработки всех N строим один сводный график для k и theta.
            comparison_figure = plot_comparison_three_methods(
                N_values=comparison_results["N"],
                method1_values=comparison_results["method_1"],
                method2_values=comparison_results["method_2"],
                method3_values=comparison_results["method_3"],
            )
            save_comparison_graph(
                figure=comparison_figure,
                comparison_results=comparison_results,
                k=k_folder,
                theta=theta,
            )


if __name__ == "__main__":
    main()
