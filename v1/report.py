import os
import numpy as np

MD_FILE = "results/results_kvantil.md"


def init_report(theta, M_outer):
    os.makedirs("results", exist_ok=True)

    with open(MD_FILE, "w") as f:
        f.write("# Experiment Results\n\n")
        f.write(f"theta={theta}, M_outer={M_outer}\n\n")

        f.write("| omega | alpha1 | alpha2 | k | N | F_mean | F_std | Fmax_mean | Fmax_std | Fmin_mean | Fmin_std | R_mean | R_std |\n")
        f.write("|------|--------|--------|---|----|--------|--------|-----------|-----------|-----------|-----------|--------|--------|\n")


def save_row(
    omega, alpha1, alpha2, k, N,
    mean_F, std_F,
    mean_Fmax, std_Fmax,
    mean_Fmin, std_Fmin,
    mean_R, std_R
):
    with open(MD_FILE, "a") as f:
        f.write(
            f"| {omega} | {alpha1} | {alpha2} | {k} | {N} | "
            f"{mean_F:.6f} | {std_F:.6f} | "
            f"{mean_Fmax:.6f} | {std_Fmax:.6f} | "
            f"{mean_Fmin:.6f} | {std_Fmin:.6f} | "
            f"{mean_R:.6f} | {std_R:.6f} |\n"
        )


# 🔥 НОВАЯ ЛОГИКА: ширина конуса
def save_local_table(R_distributions, N_values, alpha_pair, omega):
    base_folder = f"results/omega_{omega}/alpha_{alpha_pair}"
    os.makedirs(base_folder, exist_ok=True)

    file_path = f"{base_folder}/interval_width.md"

    alpha_val = (alpha_pair[0] + alpha_pair[1]) / 2

    with open(file_path, "w") as f:
        f.write("# Ширина конуса (R_high − R_low)\n\n")
        f.write(f"- ω = {omega}\n")
        f.write(f"- α = {alpha_val}\n\n")

        # заголовок
        header = "| k \\ N | " + " | ".join(map(str, N_values)) + " |\n"
        separator = "|" + "------|" * (len(N_values) + 1) + "\n"

        f.write(header)
        f.write(separator)

        for k in R_distributions:
            widths = []

            for R_list in R_distributions[k]:
                R_low = np.percentile(R_list, 5)
                R_high = np.percentile(R_list, 95)

                width = R_high - R_low
                widths.append(width)

            row = f"| {k} | " + " | ".join(f"{w:.4f}" for w in widths) + " |\n"
            f.write(row)




def save_graph_tables(results, N_values, alpha_pair, omega):
    base_folder = f"results/omega_{omega}/alpha_{alpha_pair}"
    os.makedirs(base_folder, exist_ok=True)

    alpha1, alpha2 = alpha_pair
    alpha_str = str(alpha1) if alpha1 == alpha2 else f"{alpha1}, {alpha2}"

    graphs = [
        ("F", "mean", "F_mean.md"),
        ("F", "std", "F_std.md"),
        ("F_max", "mean", "F_max_mean.md"),
        ("F_max", "std", "F_max_std.md"),
        ("F_min", "mean", "F_min_mean.md"),
        ("F_min", "std", "F_min_std.md"),
        ("R", "mean", "R_mean.md"),
        ("R", "std", "R_std.md"),
    ]

    names = {
        "F": "F(x)",
        "F_max": "F(Xmax)",
        "F_min": "F(Xmin)",
        "R": "R"
    }

    for func, metric, filename in graphs:

        with open(f"{base_folder}/{filename}", "w", encoding="utf-8") as f:

            f.write(f"# {names[func]} ({'математическое ожидание' if metric == 'mean' else 'СКО'})\n\n")

            f.write(f"- ω = {omega}\n")
            f.write(f"- α = {alpha_str}\n\n")

            header = "| k \\ N | " + " | ".join(map(str, N_values)) + " |\n"
            separator = "|" + "---|" * (len(N_values) + 1) + "\n"

            f.write(header)
            f.write(separator)

            for k in results:
                values = results[k][func][metric]

                row = (
                    f"| {k} | "
                    + " | ".join(f"{v:.6f}" for v in values)
                    + " |\n"
                )

                f.write(row)