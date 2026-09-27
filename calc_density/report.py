# report.py

import csv
import os


REPORT_FILE = "results/report.md"
REPLAYS_FILE = "results/replays.csv"


def save_replay_statistics(
    k,
    theta,
    N,
    replay_results
):
    """
    Сохраняет статистику каждого replay в CSV.

    Одна строка = один фактический эксперимент.
    """

    os.makedirs("results", exist_ok=True)

    file_exists = os.path.exists(REPLAYS_FILE)

    with open(
        REPLAYS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "k",
                "theta",
                "N",
                "replay",

                "orange_count",
                "orange_ratio",

                "blue_count",
                "blue_ratio",

                "green_count",
                "green_ratio",

                "outside_count",
                "outside_ratio",

                "total"
            ])

        for result in replay_results:
            writer.writerow([
                k,
                theta,
                N,
                result["replay"],

                result["orange_count"],
                f"{result['orange_ratio']:.12f}",

                result["blue_count"],
                f"{result['blue_ratio']:.12f}",

                result["green_count"],
                f"{result['green_ratio']:.12f}",

                result["outside_count"],
                f"{result['outside_ratio']:.12f}",

                result["total"]
            ])

    return REPLAYS_FILE


def create_report(results):
    os.makedirs("results", exist_ok=True)

    lines = [
        "# Результаты расчётов",
        "",
    ]

    for result in results:
        k = result["k"]
        theta = result["theta"]
        N = result["N"]
        m = result["m"]
        replays = result["replays"]

        areas = result["areas"]
        totals = result["replay_totals"]

        orange_boundary = (
            areas["f_min"]["right"]
        )

        blue_boundary = (
            areas["f"]["right"]
        )

        green_boundary = (
            areas["f_max"]["left"]
        )

        # Здесь в areas лежат индексы.
        # Реальные значения x берём из результата ниже
        # через сохранённые границы в replay-статистике
        # отдельно не сохраняем.

        lines.extend([
            f"## k={k}, theta={theta}, N={N}",
            "",
            f"Параметр `m = {m}`.",
            "",
            f"Количество повторений: `{replays}`.",
            "",
            "### Итог по фактическим элементам",
            "",
            "| Область | Элементов | Доля |",
            "|---|---:|---:|",
            (
                f"| Оранжевая | "
                f"{totals['orange_count']} | "
                f"{totals['orange_ratio']:.8f} |"
            ),
            (
                f"| Синяя | "
                f"{totals['blue_count']} | "
                f"{totals['blue_ratio']:.8f} |"
            ),
            (
                f"| Зелёная | "
                f"{totals['green_count']} | "
                f"{totals['green_ratio']:.8f} |"
            ),
            (
                f"| Вне областей | "
                f"{totals['outside_count']} | "
                f"{totals['outside_ratio']:.8f} |"
            ),
            (
                f"| **Всего** | "
                f"**{totals['total']}** | "
                f"**1.00000000** |"
            ),
            "",
            "### Границы областей",
            "",
            f"- Оранжевая: индекс `{orange_boundary}`",
            f"- Синяя: индекс `{blue_boundary}`",
            f"- Зелёная: индекс `{green_boundary}`",
            "",
            "### Статистика каждого replay",
            "",
            "| Replay | Оранжевая | Синяя | Зелёная | Вне областей | Всего |",
            "|---:|---:|---:|---:|---:|---:|",
        ])

        for replay in result["replay_results"]:
            lines.append(
                f"| {replay['replay']} | "
                f"{replay['orange_count']} | "
                f"{replay['blue_count']} | "
                f"{replay['green_count']} | "
                f"{replay['outside_count']} | "
                f"{replay['total']} |"
            )

        lines.append("")

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))

    return REPORT_FILE