from plotting import plot_three_densities
import os
import shutil

def main():
    if os.path.exists("results"):
        shutil.rmtree("results")
    # ==========================================
    # ПАРАМЕТРЫ
    # ==========================================

    k_values = [1, 5, 15]
    theta_values = [1]

    N_values = [10, 100]

    # ==========================================
    # ПОСТРОЕНИЕ ГРАФИКОВ
    # ==========================================

    for k in k_values:
        for theta in theta_values:
            for N in N_values:

                print(
                    f"\n"
                    f"Построение: "
                    f"k={k}, "
                    f"theta={theta}, "
                    f"N={N}"
                )

                plot_three_densities(
                    k=k,
                    theta=theta,
                    N=N
                )


if __name__ == "__main__":
    main()