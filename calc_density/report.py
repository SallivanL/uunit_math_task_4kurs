import csv
import os


RESULTS_DIR = "results"


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
    histogram,
    replay_results,
    replay_ratios,
    average_replay_ratios,
    replay_totals,
    area_statistics,
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

    save_histogram(
        figure=histogram,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_replays(
        replay_results=replay_results,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N,
    )

    save_report(
        k=k,
        theta=theta,
        N=N,
        m=m,
        replays=replays,
        x=x,
        areas=areas,
        replay_results=replay_results,
        replay_ratios=replay_ratios,
        average_replay_ratios=average_replay_ratios,
        replay_totals=replay_totals,
        area_statistics=area_statistics,
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
        f"distribution_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.png",
    )

    figure.savefig(
        graph_file,
        dpi=150,
    )

    figure.clf()


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
        f"histogram_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.png",
    )

    figure.savefig(
        histogram_file,
        dpi=150,
    )

    figure.clf()


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
            "total",
        ])

        for result in replay_results:
            writer.writerow([
                result["replay"],
                result["orange_count"],
                result["blue_count"],
                result["green_count"],
                result["total"],
            ])


def save_report(
    k,
    theta,
    N,
    m,
    replays,
    x,
    areas,
    replay_results,
    replay_ratios,
    average_replay_ratios,
    replay_totals,
    area_statistics,
    output_dir,
):
    """
    Сохраняет полный Markdown-отчёт.
    """

    report_file = os.path.join(
        output_dir,
        f"report_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.md",
    )

    orange_boundary = x[
        areas["orange"]["right"]
    ]

    green_boundary = x[
        areas["green"]["left"]
    ]

    lines = [
        "# Результаты расчёта",
        "",
        f"- `k = {k}`",
        f"- `theta = {theta}`",
        f"- `N = {N}`",
        f"- `m = {m}`",
        f"- `replays = {replays}`",
        "",
        "## Границы закрашенных областей",
        "",
        (
            "Оранжевая область определяется по распределению "
            f"минимума `f_min(x)` и содержит {m:.0%} его площади."
        ),
        "",
        (
            "Зелёная область определяется по распределению "
            f"максимума `f_max(x)` и содержит {m:.0%} его площади."
        ),
        "",
        (
            "Синяя область — весь промежуток между правой "
            "границей оранжевой области и левой границей "
            "зелёной области. Она строится по исходному "
            "распределению `f(x)`."
        ),
        "",
        "| Область | Интервал |",
        "|---|---|",
        (
            f"| Оранжевая | "
            f"`x <= {orange_boundary:.12f}` |"
        ),
        (
            f"| Синяя | "
            f"`{orange_boundary:.12f} < x < "
            f"{green_boundary:.12f}` |"
        ),
        (
            f"| Зелёная | "
            f"`x >= {green_boundary:.12f}` |"
        ),
        "",
        "## Итог по всем replay",
        "",
        "| Область | Элементов | Доля |",
        "|---|---:|---:|",
        (
            f"| Оранжевая | "
            f"{replay_totals['orange_count']} | "
            f"{replay_totals['orange_ratio']:.2%} |"
        ),
        (
            f"| Синяя | "
            f"{replay_totals['blue_count']} | "
            f"{replay_totals['blue_ratio']:.2%} |"
        ),
        (
            f"| Зелёная | "
            f"{replay_totals['green_count']} | "
            f"{replay_totals['green_ratio']:.2%} |"
        ),
        (
            f"| **Всего** | "
            f"**{replay_totals['total']}** | "
            f"**100.00%** |"
        ),
        "",
        "## Статистика областей",
        "",
        (
            "Для каждой области рассчитаны: математическое ожидание, "
            "частота, среднее количество элементов за replay, "
            "среднее квадратическое отклонение и дисперсия."
        ),
        "",
        (
            "| Область | Мат. ожидание | Частота | "
            "Среднее кол-во точек | СКО | Дисперсия |"
        ),
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Оранжевая | "
            f"{area_statistics['orange']['mathematical_expectation']:.6f} | "
            f"{area_statistics['orange']['frequency']:.2%} | "
            f"{area_statistics['orange']['mean']:.6f} | "
            f"{area_statistics['orange']['standard_deviation']:.6f} | "
            f"{area_statistics['orange']['variance']:.6f} |"
        ),
        (
            f"| Синяя | "
            f"{area_statistics['blue']['mathematical_expectation']:.6f} | "
            f"{area_statistics['blue']['frequency']:.2%} | "
            f"{area_statistics['blue']['mean']:.6f} | "
            f"{area_statistics['blue']['standard_deviation']:.6f} | "
            f"{area_statistics['blue']['variance']:.6f} |"
        ),
        (
            f"| Зелёная | "
            f"{area_statistics['green']['mathematical_expectation']:.6f} | "
            f"{area_statistics['green']['frequency']:.2%} | "
            f"{area_statistics['green']['mean']:.6f} | "
            f"{area_statistics['green']['standard_deviation']:.6f} | "
            f"{area_statistics['green']['variance']:.6f} |"
        ),
        "",
        "## Средняя доля точек по областям",
        "",
        (
            f"Значения `n / N` усреднены по всем "
            f"{replays} replay."
        ),
        "",
        (
            "| Область | Среднее n/N | "
            "Среднее количество точек |"
        ),
        "|---|---:|---:|",
        (
            f"| Область 1 | "
            f"{average_replay_ratios['orange_ratio']:.6f} | "
            f"{average_replay_ratios['orange_ratio'] * N:.6f} |"
        ),
        (
            f"| Область 2 | "
            f"{average_replay_ratios['blue_ratio']:.6f} | "
            f"{average_replay_ratios['blue_ratio'] * N:.6f} |"
        ),
        (
            f"| Область 3 | "
            f"{average_replay_ratios['green_ratio']:.6f} | "
            f"{average_replay_ratios['green_ratio'] * N:.6f} |"
        ),
        (
            f"| **Сумма** | "
            f"**{average_replay_ratios['total']:.6f}** | "
            f"**{average_replay_ratios['total'] * N:.6f}** |"
        ),
        "",
        "## Доли точек по каждому replay",
        "",
        (
            "Для проверки приведены значения `n / N` "
            "для каждого replay."
        ),
        "",
        (
            "| Replay | Область 1 (n1/N) | "
            "Область 2 (n2/N) | Область 3 (n3/N) | "
            "Сумма |"
        ),
        "|---:|---:|---:|---:|---:|",
    ]

    for ratio in replay_ratios:
        ratio_sum = (
            ratio["orange_ratio"]
            + ratio["blue_ratio"]
            + ratio["green_ratio"]
        )

        lines.append(
            f"| {ratio['replay']} | "
            f"{ratio['orange_ratio']:.6f} | "
            f"{ratio['blue_ratio']:.6f} | "
            f"{ratio['green_ratio']:.6f} | "
            f"{ratio_sum:.6f} |"
        )

    lines.extend([
        "",
        "## Статистика каждого replay",
        "",
        (
            "| Replay | Оранжевая | Синяя | "
            "Зелёная | Всего |"
        ),
        "|---:|---:|---:|---:|---:|",
    ])

    for replay in replay_results:
        lines.append(
            f"| {replay['replay']} | "
            f"{replay['orange_count']} | "
            f"{replay['blue_count']} | "
            f"{replay['green_count']} | "
            f"{replay['total']} |"
        )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )