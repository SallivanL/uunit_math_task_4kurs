import os
import shutil

from calc import calculate_densities, calculate_areas
from logger import logger
from plotting import plot_three_densities
from points import save_plot_points
from report import create_report


def main():
    # Параметры эксперимента
    m = 0.8

    k_values = [1, 5, 15]
    theta_values = [1]
    N_values = [10, 100]

    # Подготовка результатов
    if os.path.exists("results"):
        shutil.rmtree("results")

    os.makedirs("results", exist_ok=True)

    results = []

    for k in k_values:
        for theta in theta_values:
            for N in N_values:
                logger.info("Расчёт: k=%s, theta=%s, N=%s, m=%s", k, theta, N, m)

                x, f, f_min, f_max = calculate_densities(k=k, theta=theta, N=N)

                areas = calculate_areas(x=x, f=f, f_min=f_min, f_max=f_max, m=m)

                save_plot_points(x=x, f=f, f_min=f_min, f_max=f_max, k=k, theta=theta, N=N)

                plot_three_densities(
                    x=x,
                    f=f,
                    f_min=f_min,
                    f_max=f_max,
                    areas=areas,
                    k=k,
                    theta=theta,
                    N=N,
                )

                results.append(
                    {
                        "k": k,
                        "theta": theta,
                        "N": N,
                        "m": m,
                        "areas": areas,
                    }
                )

    report_file = create_report(results)

    logger.info("Расчёты завершены. Отчёт: %s", report_file)


if __name__ == "__main__":
    main()
