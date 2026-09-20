import matplotlib.pyplot as plt
import os
import numpy as np


def plot_results(results, N_values, alpha_pair, omega):
    base_folder = f"results/omega_{omega}/alpha_{alpha_pair}"
    os.makedirs(base_folder, exist_ok=True)

    functions = ["F", "F_max", "F_min", "R"]

    line_styles = ['-', '--', '-.', ':']

    for func in functions:
        plt.figure()

        for idx, k in enumerate(results):
            plt.plot(
                N_values,
                results[k][func]["mean"],
                linestyle=line_styles[idx % len(line_styles)],
                color="black",
                label=f"k={k}"
            )

        alpha_str = format_alpha(alpha_pair)
        plt.title(f"{func} mean (α={alpha_str}, ω={omega})")
        plt.xlabel("N")
        plt.legend()
        plt.grid()

        plt.savefig(f"{base_folder}/{func}_mean.png")
        plt.close()

        plt.figure()

        for idx, k in enumerate(results):
            plt.plot(
                N_values,
                results[k][func]["std"],
                linestyle=line_styles[idx % len(line_styles)],
                color="black",
                label=f"k={k}"
            )

        plt.title(f"{func} std")
        plt.xlabel("N")
        plt.legend()
        plt.grid()

        plt.savefig(f"{base_folder}/{func}_std.png")
        plt.close()


def plot_cone(R_distributions, N_values, alpha_pair, omega):
    base_folder = f"results/omega_{omega}/alpha_{alpha_pair}"
    os.makedirs(base_folder, exist_ok=True)

    plt.figure()

    line_styles = ['-', '--', '-.', ':']

    for idx, k in enumerate(R_distributions):
        R_low = []
        R_high = []

        for R_list in R_distributions[k]:
            R_low.append(np.percentile(R_list, 5))
            R_high.append(np.percentile(R_list, 95))

        style = line_styles[idx % len(line_styles)]

        # границы
        plt.plot(
            N_values,
            R_low,
            linestyle=style,
            color="black",
            linewidth=1
        )

        plt.plot(
            N_values,
            R_high,
            linestyle=style,
            color="black",
            linewidth=1,
            label=f"k={k}"
        )

        # заливка
        plt.fill_between(
            N_values,
            R_low,
            R_high,
            color="black",
            alpha=0.1
        )

    plt.axhline(1.0, linestyle=":", color="black")

    alpha_str = format_alpha(alpha_pair)
    plt.title(f"(α={alpha_str}, ω={omega})")

    plt.xlabel("N")
    plt.ylabel("R")

    plt.legend()
    plt.grid()

    plt.savefig(f"{base_folder}/R_cone.png")
    plt.close()


def format_alpha(alpha_pair):
    a1, a2 = alpha_pair
    if a1 == a2:
        return f"{a1}"
    return f"({a1}, {a2})"