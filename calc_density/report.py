import csv
import math
import os
import shutil
import matplotlib.pyplot as plt


RESULTS_DIR = "results"


def clear_results():
    """Удаляет предыдущие результаты экспериментов перед новым запуском."""
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)


def get_experiment_dir(k, N):
    """
    Возвращает и создаёт директорию конкретного эксперимента.

    results/
        k_X/
            N_Y/
    """

    directory = os.path.join(
        RESULTS_DIR,
        f"k_{k}",
        f"N_{N}",
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    return directory


def save_experiment(
    k,
    theta,
    N,
    m,
    replays,
    x,
    areas,
    figure,
    cdf_figure,
    histogram,
    replay_histogram,
    method2_replay_histogram,
    cdf_histogram,
    equal_cdf_figure,
    equal_cdf_histogram,
    equal_cdf_areas,
    method2_replay_totals,
    method1_chi_squared,
    method2_chi_squared,
    replay_results,
    replay_ratios,
    average_replay_ratios,
    replay_totals,
    area_statistics,
    cdf_area_probabilities,
    method3_distribution_figure,
    method3_cdf_figure,
    method3_theoretical_histogram,
    method3_replay_histogram,
    method3_areas,
    method3_replay_totals,
    method3_chi_squared,
    k_display=None,
    sample_counts=None,
):
    """
    Полностью сохраняет один эксперимент.
    """

    output_dir = get_experiment_dir(
        k=k,
        N=N,
    )

    save_graph(
        figure=figure,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_cdf_graph(
        figure=cdf_figure,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_histogram(
        figure=histogram,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_cdf_histogram(
        figure=cdf_histogram,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_replay_histogram(
        figure=replay_histogram,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_method2_replay_histogram(
        figure=method2_replay_histogram,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_equal_cdf_graph(
        figure=equal_cdf_figure, output_dir=output_dir,
        k=k, theta=theta, N=N,
    )
    save_equal_cdf_histogram(
        figure=equal_cdf_histogram, output_dir=output_dir,
        k=k, theta=theta, N=N,
    )

    # Метод 3 по логике расчёта; для его графиков задан префикс method_3.
    save_method3_distribution_graph(
        figure=method3_distribution_figure, output_dir=output_dir,
        k=k, theta=theta, N=N,
    )
    save_method3_cdf_graph(
        figure=method3_cdf_figure, output_dir=output_dir,
        k=k, theta=theta, N=N,
    )
    save_method3_theoretical_histogram(
        figure=method3_theoretical_histogram, output_dir=output_dir,
        k=k, theta=theta, N=N,
    )
    save_method3_replay_histogram(
        figure=method3_replay_histogram, output_dir=output_dir,
        k=k, theta=theta, N=N,
    )

    save_replays(
        replay_results=replay_results,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_chi_squared_comparison(
        method1_chi_squared=method1_chi_squared,
        method2_chi_squared=method2_chi_squared,
        method3_chi_squared=method3_chi_squared,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_report(
        k=k,
        theta=theta,
        N=N,
        replays=replays,
        method1_chi_squared=method1_chi_squared,
        method2_chi_squared=method2_chi_squared,
        method3_chi_squared=method3_chi_squared,
        method3_areas=method3_areas,
        output_dir=output_dir,
        k_display=k_display,
        sample_counts=sample_counts,
    )


def save_graph(
    figure,
    output_dir,
    k,
    theta,
    N,
):
    """
    Сохраняет основной график распределений.
    """

    graph_file = os.path.join(
        output_dir,
        f"initial_graphs_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.png",
    )

    figure.savefig(
        graph_file,
        dpi=150,
    )

    plt.close(figure)


def save_cdf_graph(
    figure,
    output_dir,
    k,
    theta,
    N,
):
    """
    Сохраняет график F(x) с отмеченными границами.
    """

    cdf_file = os.path.join(
        output_dir,
        f"method_1_cdf_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.png",
    )

    figure.savefig(
        cdf_file,
        dpi=150,
    )

    plt.close(figure)


def save_histogram(
    figure,
    output_dir,
    k,
    theta,
    N,
):
    """
    Сохраняет итоговую гистограмму
    средних долей по трём областям.
    """

    histogram_file = os.path.join(
        output_dir,
        f"initial_histogram_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.png",
    )

    figure.savefig(
        histogram_file,
        dpi=150,
    )

    plt.close(figure)


def save_cdf_histogram(
    figure,
    output_dir,
    k,
    theta,
    N,
):
    """
    Сохраняет гистограмму теоретических вероятностей,
    рассчитанных через F(x).
    """

    histogram_file = os.path.join(
        output_dir,
        f"method_1_theoretical_probabilities_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.png",
    )

    figure.savefig(
        histogram_file,
        dpi=150,
    )

    plt.close(figure)



def save_replay_histogram(figure, output_dir, k, theta, N):
    """Сохраняет единый рисунок со всеми replay-гистограммами."""
    filename = os.path.join(
        output_dir,
        f"method_1_replay_histograms_k_{k}_theta_{theta}_N_{N}.png",
    )
    figure.savefig(filename, dpi=150)
    plt.close(figure)


def save_method2_replay_histogram(figure, output_dir, k, theta, N):
    """Сохраняет единый рисунок со всеми replay-гистограммами метода 2."""
    filename = os.path.join(
        output_dir,
        f"method_2_replay_histograms_k_{k}_theta_{theta}_N_{N}.png",
    )
    figure.savefig(filename, dpi=150)
    plt.close(figure)


def save_equal_cdf_graph(figure, output_dir, k, theta, N):
    filename = os.path.join(output_dir, f"method_2_cdf_k_{k}_theta_{theta}_N_{N}.png")
    figure.savefig(filename, dpi=150)
    plt.close(figure)


def save_equal_cdf_histogram(figure, output_dir, k, theta, N):
    filename = os.path.join(output_dir, f"method_2_theoretical_probabilities_k_{k}_theta_{theta}_N_{N}.png")
    figure.savefig(filename, dpi=150)
    plt.close(figure)



def save_method3_distribution_graph(figure, output_dir, k, theta, N):
    filename = os.path.join(
        output_dir,
        f"method_3_distribution_k_{k}_theta_{theta}_N_{N}.png",
    )
    figure.savefig(filename, dpi=150)
    plt.close(figure)


def save_method3_cdf_graph(figure, output_dir, k, theta, N):
    filename = os.path.join(
        output_dir,
        f"method_3_cdf_k_{k}_theta_{theta}_N_{N}.png",
    )
    figure.savefig(filename, dpi=150)
    plt.close(figure)


def save_method3_theoretical_histogram(figure, output_dir, k, theta, N):
    filename = os.path.join(
        output_dir,
        f"method_3_theoretical_probabilities_k_{k}_theta_{theta}_N_{N}.png",
    )
    figure.savefig(filename, dpi=150)
    plt.close(figure)


def save_method3_replay_histogram(figure, output_dir, k, theta, N):
    filename = os.path.join(
        output_dir,
        f"method_3_replay_histograms_k_{k}_theta_{theta}_N_{N}.png",
    )
    figure.savefig(filename, dpi=150)
    plt.close(figure)

def save_replays(
    replay_results,
    output_dir,
    k,
    theta,
    N,
):
    """
    Сохраняет результаты каждого replay в CSV.
    """

    filename = os.path.join(
        output_dir,
        f"replays_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.csv",
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        base_columns = [
            "replay",
            "orange_count",
            "blue_count",
            "green_count",
            "method2_orange_count",
            "method2_blue_count",
            "method2_green_count",
            "method3_orange_count",
            "method3_blue_count",
            "method3_green_count",
            "total",
        ]
        quota_columns = sorted({
            key for result in replay_results for key in result
            if key.startswith("selected_from_k_")
        })
        writer.writerow(base_columns + quota_columns)

        for result in replay_results:
            row = [
                result["replay"],
                result["orange_count"],
                result["blue_count"],
                result["green_count"],
                result.get("method2_orange_count", ""),
                result.get("method2_blue_count", ""),
                result.get("method2_green_count", ""),
                result.get("method3_orange_count", ""),
                result.get("method3_blue_count", ""),
                result.get("method3_green_count", ""),
                result["total"],
            ]
            writer.writerow(row + [result.get(key, "") for key in quota_columns])


def save_chi_squared_comparison(
    method1_chi_squared,
    method2_chi_squared,
    method3_chi_squared,
    output_dir,
    k,
    theta,
    N,
):
    """Сохраняет построчную детализацию расчёта chi² для всех методов."""
    filename = os.path.join(
        output_dir,
        f"chi_squared_comparison_k_{k}_theta_{theta}_N_{N}.csv",
    )
    area_labels = {
        "orange": "Оранжевая",
        "blue": "Синяя",
        "green": "Зелёная",
    }
    methods = (
        ("Метод 1", method1_chi_squared),
        ("Метод 2", method2_chi_squared),
        ("Метод 3", method3_chi_squared),
    )

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "method",
            "area",
            "theoretical_probability_p",
            "empirical_probability_p_hat",
            "observed_count",
            "total_count",
            "chi_squared_contribution",
            "count_difference_observed_minus_theoretical",
            "relative_deviation_percent",
            "chi_squared_total",
        ])
        for method_name, result in methods:
            for area_name in ("orange", "blue", "green"):
                area = result["areas"][area_name]
                theoretical_probability = area["theoretical_probability"]
                empirical_probability = area["empirical_probability"]
                deviation_percent = (
                    (empirical_probability - theoretical_probability)
                    / theoretical_probability
                    * 100
                    if theoretical_probability > 0
                    else None
                )
                writer.writerow([
                    method_name,
                    area_labels[area_name],
                    f"{theoretical_probability:.12g}",
                    f"{empirical_probability:.12g}",
                    area["observed_count"],
                    area["total_count"],
                    f"{area['contribution']:.12g}",
                    math.ceil(area["observed_count"] - theoretical_probability * area["total_count"]),
                    f"{deviation_percent:.12g}" if deviation_percent is not None else "",
                    f"{result['chi_squared']:.12g}",
                ])


def save_report(
    k,
    theta,
    N,
    replays,
    method1_chi_squared,
    method2_chi_squared,
    method3_chi_squared,
    method3_areas,
    output_dir,
    k_display=None,
    sample_counts=None,
):
    """Сохраняет Markdown-отчёт по всем трём методам."""
    report_file = os.path.join(
        output_dir,
        f"report_k_{k}_theta_{theta}_N_{N}.md",
    )

    methods = (
        ("Метод 1", method1_chi_squared),
        ("Метод 2", method2_chi_squared),
        ("Метод 3", method3_chi_squared),
    )
    area_labels = {
        "orange": "Оранжевая",
        "blue": "Синяя",
        "green": "Зелёная",
    }
    area_order = ("orange", "blue", "green")

    lines = [
        "# Результаты эксперимента",
        "",
        f"`k = {k_display if k_display is not None else k}`, `θ = {theta}`, `N = {N}`, `replays = {replays}`",
        "",
    ]
    if sample_counts is not None:
        quota_text = ", ".join(
            f"k={component_k}: {count} элементов"
            for component_k, count in zip(k_display, sample_counts)
        )
        lines.extend([f"Квоты смешанной выборки: {quota_text}.", ""])
    lines.extend([
        "## 1. Результаты χ²",
        "",
        "Формула: `χ² = Σ ((p_j - p̂_j) / p_j)²`, где `p_j` — теоретическая вероятность, `p̂_j` — экспериментальная вероятность.",
        "",
        "| Метод | Область | Теоретическая вероятность p_j | Экспериментальная вероятность p̂_j | Вклад в χ² |",
        "|---|---|---:|---:|---:|",
    ])

    for method_label, result in methods:
        for area_name in area_order:
            area = result["areas"][area_name]
            lines.append(
                f"| {method_label} | {area_labels[area_name]} | "
                f"{area['theoretical_probability']:.8f} | "
                f"{area['empirical_probability']:.8f} | "
                f"{area['contribution']:.10f} |"
            )
        lines.append(
            f"| **{method_label}: итог χ²** | — | — | — | "
            f"**{result['chi_squared']:.10f}** |"
        )

    lines.extend([
        "",
        "## 2. Сравнение теоретического и фактического количества точек",
        "",
        f"Теоретическое количество рассчитано для всех {replays} повторений: `p_j × N × replays`. Фактическое количество — число элементов, попавших в соответствующую область во всех повторениях. Отклонение в процентах: `(факт − теория) / теория × 100%`; знак `+` означает превышение теории, знак `−` — значение ниже теории.",
        "",
        "| Метод | Область | Теоретическое количество точек | Фактическое количество точек | Разница (факт − теория), точек | Отклонение от теории, % |",
        "|---|---|---:|---:|---:|---:|",
    ])

    for method_label, result in methods:
        total_count = int(result["total_count"])
        for area_name in area_order:
            area = result["areas"][area_name]
            theoretical_count = (
                area["theoretical_probability"] * total_count
            )
            observed_count = area["observed_count"]
            deviation_percent = (
                (observed_count - theoretical_count) / theoretical_count * 100
                if theoretical_count > 0
                else None
            )
            count_difference = math.ceil(observed_count - theoretical_count)
            deviation_text = (
                f"{deviation_percent:+.2f}%"
                if deviation_percent is not None
                else "н/д"
            )
            lines.append(
                f"| {method_label} | {area_labels[area_name]} | "
                f"{theoretical_count:.4f} | {observed_count} | "
                f"{count_difference:+d} | {deviation_text} |"
            )
        lines.append(
            f"| **{method_label}: всего** | — | **{total_count:.4f}** | "
            f"**{total_count}** | **+0** | **+0.00%** |"
        )

    lines.append("")

    with open(report_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_comparison_graph(figure, comparison_results, k, theta):
    """Сохраняет сводный график и CSV с данными в корень results/k_X/."""
    expected_keys = ("method_1", "method_2", "method_3")
    if "N" not in comparison_results:
        raise ValueError("В comparison_results отсутствует список N")
    row_count = len(comparison_results["N"])
    if row_count == 0:
        raise ValueError("comparison_results не содержит значений N")
    if any(
        key not in comparison_results
        or len(comparison_results[key]) != row_count
        for key in expected_keys
    ):
        raise ValueError("Количество значений методов должно совпадать с количеством N")

    output_dir = os.path.join(RESULTS_DIR, f"k_{k}")
    os.makedirs(output_dir, exist_ok=True)

    filename_suffix = f"theta_{theta}"
    graph_file = os.path.join(
        output_dir,
        f"comparison_three_methods_{filename_suffix}.png",
    )
    figure.savefig(graph_file, dpi=150)
    plt.close(figure)

    csv_file = os.path.join(
        output_dir,
        f"comparison_three_methods_{filename_suffix}.csv",
    )
    with open(csv_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "N",
            "method_1_expected_squared_error_M_X2",
            "method_2_expected_squared_error_M_X2",
            "method_3_expected_squared_error_M_X2",
        ])

        for index in range(row_count):
            writer.writerow([
                comparison_results["N"][index],
                f"{comparison_results['method_1'][index]:.12g}",
                f"{comparison_results['method_2'][index]:.12g}",
                f"{comparison_results['method_3'][index]:.12g}",
            ])
