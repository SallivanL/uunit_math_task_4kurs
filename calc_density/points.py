import os

from logger import log


def save_plot_points(
    x,
    f,
    phi_max,
    psi_min,
    k,
    theta,
    N,
    save_points=100
):
    """
    Сохраняет выбранные точки графика в общий log.md.
    """

    indices = [
        round(i * (len(x) - 1) / (save_points - 1))
        for i in range(save_points)
    ]

    lines = [
        f"## Точки графика: k={k}, theta={theta}, N={N}",
        "",
        "| x | f(x) | phi_max(x) | psi_min(x) |",
        "|---:|---:|---:|---:|"
    ]

    for i in indices:
        lines.append(
            f"| {x[i]:.12f} | "
            f"{f[i]:.12f} | "
            f"{phi_max[i]:.12f} | "
            f"{psi_min[i]:.12f} |"
        )

    text = "\n".join(lines) + "\n"

    os.makedirs("results", exist_ok=True)

    with open(
        "results/log.md",
        "a",
        encoding="utf-8"
    ) as file:
        file.write(text)

    return "results/log.md"