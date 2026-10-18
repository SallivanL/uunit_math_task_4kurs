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
        output_dir=output_dir,
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

        writer.writerow([
            "replay",
            "orange_count",
            "blue_count",
            "green_count",
            "method2_orange_count",
            "method2_blue_count",
            "method2_green_count",
            "total",
        ])

        for result in replay_results:
            writer.writerow([
                result["replay"],
                result["orange_count"],
                result["blue_count"],
                result["green_count"],
                result.get("method2_orange_count", ""),
                result.get("method2_blue_count", ""),
                result.get("method2_green_count", ""),
                result["total"],
            ])


def save_chi_squared_comparison(
    method1_chi_squared,
    method2_chi_squared,
    output_dir,
    k,
    theta,
    N,
):
    """Сохраняет построчную детализацию расчёта chi² для двух методов."""
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
    output_dir,
):
    """Сохраняет краткий Markdown-отчёт только с двумя сравнениями."""
    report_file = os.path.join(
        output_dir,
        f"report_k_{k}_theta_{theta}_N_{N}.md",
    )

    methods = (
        ("Метод 1", method1_chi_squared),
        ("Метод 2", method2_chi_squared),
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
        f"`k = {k}`, `θ = {theta}`, `N = {N}`, `replays = {replays}`",
        "",
        "## 1. Результаты χ²",
        "",
        "Формула: `χ² = Σ ((p_j - p̂_j) / p_j)²`, где `p_j` — теоретическая вероятность, `p̂_j` — экспериментальная вероятность.",
        "",
        "| Метод | Область | Теоретическая вероятность p_j | Экспериментальная вероятность p̂_j | Вклад в χ² |",
        "|---|---|---:|---:|---:|",
    ]

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
