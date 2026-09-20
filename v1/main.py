import numpy as np
from calc import run_experiment
from plotting import plot_results, plot_cone
from report import (
    init_report,
    save_row,
    save_local_table,
    save_graph_tables,
)

def main():
    theta = 1
    M_outer = 600

    k_values = [1, 2, 3, 4, 5]
    N_values = [10,20,30,40,50,60,70,80,90,100]

    alpha_values = [
        (0, 0),
        (0.5, 0.5),
        (1, 1),
        (0, 0.5),
        (0.5, 1),
    ]

    omega_values = [
        0,
        0.05,
        0.1,
        0.2
    ]

    init_report(theta, M_outer)

    for omega in omega_values:
        print(f"\n================ OMEGA = {omega} ================")

        for alpha_pair in alpha_values:
            print(f"\n================ alpha = {alpha_pair} ================")

            results = {}
            R_distributions = {}  # 🔥 для конуса

            for k in k_values:
                print(f"\n========== omega={omega} | alpha={alpha_pair} | k={k} ==========")

                results[k] = {
                    "F": {"mean": [], "std": []},
                    "F_max": {"mean": [], "std": []},
                    "F_min": {"mean": [], "std": []},
                    "R": {"mean": [], "std": []},
                }

                R_distributions[k] = []  # 🔥 список списков (по каждому N)

                for N in N_values:
                    print(f"[ω={omega}] [α={alpha_pair}] [k={k}] N={N} -> запуск...")

                    D_F_list = []
                    D_Fmax_list = []
                    D_Fmin_list = []
                    R_list = []

                    for _ in range(M_outer):
                        D_F, D_Fmax, D_Fmin, R = run_experiment(
                            k, theta, N, alpha_pair, omega
                        )

                        D_F_list.append(D_F)
                        D_Fmax_list.append(D_Fmax)
                        D_Fmin_list.append(D_Fmin)
                        R_list.append(R)

                    # 🔥 сохраняем распределение R для конуса
                    R_distributions[k].append(R_list)

                    mean_F = np.mean(D_F_list)
                    std_F = np.std(D_F_list)

                    mean_Fmax = np.mean(D_Fmax_list)
                    std_Fmax = np.std(D_Fmax_list)

                    mean_Fmin = np.mean(D_Fmin_list)
                    std_Fmin = np.std(D_Fmin_list)

                    mean_R = np.mean(R_list)
                    std_R = np.std(R_list)

                    results[k]["F"]["mean"].append(mean_F)
                    results[k]["F"]["std"].append(std_F)

                    results[k]["F_max"]["mean"].append(mean_Fmax)
                    results[k]["F_max"]["std"].append(std_Fmax)

                    results[k]["F_min"]["mean"].append(mean_Fmin)
                    results[k]["F_min"]["std"].append(std_Fmin)

                    results[k]["R"]["mean"].append(mean_R)
                    results[k]["R"]["std"].append(std_R)

                    print(
                        f"[ω={omega}] [α={alpha_pair}] [k={k}] N={N} DONE | "
                        f"F=({mean_F:.5f}, {std_F:.5f}) | "
                        f"Fmax=({mean_Fmax:.5f}, {std_Fmax:.5f}) | "
                        f"Fmin=({mean_Fmin:.5f}, {std_Fmin:.5f}) | "
                        f"R=({mean_R:.5f}, {std_R:.5f})"
                    )

                    save_row(
                        omega, alpha_pair[0], alpha_pair[1], k, N,
                        mean_F, std_F,
                        mean_Fmax, std_Fmax,
                        mean_Fmin, std_Fmin,
                        mean_R, std_R
                    )


            plot_results(results, N_values, alpha_pair, omega)
            plot_cone(R_distributions, N_values, alpha_pair, omega)

            save_local_table(R_distributions, N_values, alpha_pair, omega)
            save_graph_tables(results, N_values, alpha_pair, omega)
if __name__ == "__main__":
    main()