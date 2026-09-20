import os


REPORT_FILE = "results/report.md"


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

        lines.extend([
            f"## k={k}, theta={theta}, N={N}",
            "",
            f"Параметр `m = {m}`.",
            "",
            "| Функция | Площадь | Общая площадь | Доля |",
            "|---|---:|---:|---:|",
        ])

        for name in ("f", "f_min", "f_max"):
            area = result["areas"][name]

            lines.append(
                f"| `{name}` | "
                f"{area['area']:.8f} | "
                f"{area['total_area']:.8f} | "
                f"{area['ratio']:.8f} |"
            )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))

    return REPORT_FILE