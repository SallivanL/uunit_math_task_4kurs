# report.py

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
        f"N_{N}"
    )

    os.makedirs(
        directory,
        exist_ok=True
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
    points,
    figure,
    replay_results,
    replay_totals
):
    """
    Полностью сохраняет один эксперимент.

    Именно эта функция отвечает за все файлы.
    """

    output_dir = get_experiment_dir(
        k=k,
        N=N
    )

    save_graph(
        figure=figure,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N
    )

    save_points(
        points=points,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N
    )

    save_replays(
        replay_results=replay_results,
        output_dir=output_dir,
        k=k,
        theta=theta,
        N=N
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
        replay_totals=replay_totals,
        output_dir=output_dir
    )


def save_graph(
    figure,
    output_dir,
    k,
    theta,
    N
):
    graph_file = os.path.join(
        output_dir,
        f"distribution_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.png"
    )

    figure.savefig(
        graph_file,
        dpi=150
    )

    # Закрываем figure после сохранения.
    figure.clf()


def save_points(
    points,
    output_dir,
    k,
    theta,
    N
):
    points_file = os.path.join(
        output_dir,
        f"points_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.md"
    )

    lines = [
        f"# Точки графика",
        "",
        f"- `k = {k}`",
        f"- `theta = {theta}`",
        f"- `N = {N}`",
        "",
        "| x | f(x) | f_min(x) | f_max(x) |",
        "|---:|---:|---:|---:|"
    ]

    for point in points["points"]:
        lines.append(
            f"| {point['x']:.12f} | "
            f"{point['f']:.12f} | "
            f"{point['f_min']:.12f} | "
            f"{point['f_max']:.12f} |"
        )

    with open(
        points_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(lines)
        )


def save_replays(replay_results, output_dir, k, theta, N):
    filename = os.path.join(
        output_dir,
        f"replays_k_{k}_theta_{theta}_N_{N}.csv"
    )

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "replay",
            "orange_count",
            "blue_count",
            "green_count",
            "outside_count",
            "total",
        ])

        for result in replay_results:
            writer.writerow([
                result["replay"],
                result["orange_count"],
                result["blue_count"],
                result["green_count"],
                result["outside_count"],
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
    replay_totals,
    output_dir
):
    report_file = os.path.join(
        output_dir,
        f"report_"
        f"k_{k}_"
        f"theta_{theta}_"
        f"N_{N}.md"
    )

    orange_boundary = x[
        areas["f_min"]["right"]
    ]

    blue_boundary = x[
        areas["f"]["right"]
    ]

    green_boundary = x[
        areas["f_max"]["left"]
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
        "| Область | Интервал |",
        "|---|---|",
        (
            f"| Оранжевая | "
            f"`x <= {orange_boundary:.12f}` |"
        ),
        (
            f"| Синяя | "
            f"`{orange_boundary:.12f} < x <= "
            f"{blue_boundary:.12f}` |"
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
            f"| Вне областей | "
            f"{replay_totals['outside_count']} | "
            f"{replay_totals['outside_ratio']:.2%} |"
        ),
        (
            f"| **Всего** | "
            f"**{replay_totals['total']}** | "
            f"**100.00%** |"
        ),
        "",
        "## Статистика каждого replay",
        "",
        "| Replay | Оранжевая | Синяя | Зелёная | Вне областей | Всего |",
        "|---:|---:|---:|---:|---:|---:|"
    ]

    for replay in replay_results:
        lines.append(
            f"| {replay['replay']} | "
            f"{replay['orange_count']} | "
            f"{replay['blue_count']} | "
            f"{replay['green_count']} | "
            f"{replay['outside_count']} | "
            f"{replay['total']} |"
        )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(lines)
        )