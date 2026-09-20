import os


def save_plot_points(
    x,
    f,
    f_min,
    f_max,
    k,
    theta,
    N,
    save_points=100
):
    indices = [
        round(i * (len(x) - 1) / (save_points - 1))
        for i in range(save_points)
    ]

    lines = [
        f"## Точки графика: k={k}, theta={theta}, N={N}",
        "",
        "| x | f(x) | f_min(x) | f_max(x) |",
        "|---:|---:|---:|---:|"
    ]

    for i in indices:
        lines.append(
            f"| {x[i]:.12f} | "
            f"{f[i]:.12f} | "
            f"{f_min[i]:.12f} | "
            f"{f_max[i]:.12f} |"
        )

    os.makedirs("results", exist_ok=True)

    points_file = "results/points.md"

    with open(
        points_file,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(lines) + "\n\n"
        )

    return points_file