# main.py

import os
import shutil

import numpy as np

from calc import (
    calculate_densities,
    calculate_areas,
    run_replays,
    calculate_replay_totals,
)
from logger import logger
from plotting import plot_three_densities
from points import save_plot_points
from report import (
    create_report,
    save_replay_statistics,
)


def main():
    # Параметры эксперимента
    m = 0.8 # область на графике
    replays = 50 # Повторы эксперимента

    k_values = [1, 8] # Параметр формы

    theta_values = [1] # Параметр масштаба
    N_values = [30] # Кол-во элементов выборки

    # None означает, что каждый запуск программы
    # будет генерировать новую случайную последовательность.
    random_seed = None

    rng = np.random.default_rng(random_seed)

    # Подготовка результатов
    if os.path.exists("results"):
        shutil.rmtree("results")

    os.makedirs("results", exist_ok=True)

    results = []

    for k in k_values:
        for theta in theta_values:
            for N in N_values:
                logger.info(
                    "Расчёт: k=%s, theta=%s, N=%s, m=%s, replays=%s",
                    k,
                    theta,
                    N,
                    m,
                    replays
                )

                # Теоретические распределения
                x, f, f_min, f_max = calculate_densities(
                    k=k,
                    theta=theta,
                    N=N
                )

                # Границы закрашенных областей
                areas = calculate_areas(
                    x=x,
                    f=f,
                    f_min=f_min,
                    f_max=f_max,
                    m=m
                )

                # Сохраняем точки для графика
                save_plot_points(
                    x=x,
                    f=f,
                    f_min=f_min,
                    f_max=f_max,
                    k=k,
                    theta=theta,
                    N=N
                )

                # Строим график
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

                # ==================================================
                # ФАКТИЧЕСКИЙ ЭКСПЕРИМЕНТ
                # ==================================================

                replay_results = run_replays(
                    k=k,
                    theta=theta,
                    N=N,
                    replays=replays,
                    x=x,
                    areas=areas,
                    rng=rng
                )

                replay_totals = calculate_replay_totals(
                    replay_results
                )

                # Сохраняем статистику каждого replay
                save_replay_statistics(
                    k=k,
                    theta=theta,
                    N=N,
                    replay_results=replay_results
                )

                # Вывод результатов в консоль
                orange_boundary = (
                    x[areas["f_min"]["right"]]
                )

                blue_boundary = (
                    x[areas["f"]["right"]]
                )

                green_boundary = (
                    x[areas["f_max"]["left"]]
                )

                print()
                print("=" * 70)
                print(
                    f"k={k}, theta={theta}, "
                    f"N={N}, replays={replays}"
                )
                print("=" * 70)

                print("Границы областей:")
                print(
                    f"  Оранжевая: x <= "
                    f"{orange_boundary:.8f}"
                )
                print(
                    f"  Синяя:     "
                    f"{orange_boundary:.8f} < x <= "
                    f"{blue_boundary:.8f}"
                )
                print(
                    f"  Зелёная:   x >= "
                    f"{green_boundary:.8f}"
                )

                print()
                print("Итог по всем replay:")
                print(
                    f"  Всего элементов: "
                    f"{replay_totals['total']}"
                )

                print(
                    f"  Оранжевая: "
                    f"{replay_totals['orange_count']} "
                    f"({replay_totals['orange_ratio']:.4%})"
                )

                print(
                    f"  Синяя:     "
                    f"{replay_totals['blue_count']} "
                    f"({replay_totals['blue_ratio']:.4%})"
                )

                print(
                    f"  Зелёная:   "
                    f"{replay_totals['green_count']} "
                    f"({replay_totals['green_ratio']:.4%})"
                )

                print(
                    f"  Вне областей: "
                    f"{replay_totals['outside_count']} "
                    f"({replay_totals['outside_ratio']:.4%})"
                )

                print()
                print("Статистика отдельных replay:")

                for replay_result in replay_results:
                    print(
                        f"  Replay "
                        f"{replay_result['replay']:3d}: "
                        f"orange="
                        f"{replay_result['orange_count']:2d}, "
                        f"blue="
                        f"{replay_result['blue_count']:2d}, "
                        f"green="
                        f"{replay_result['green_count']:2d}, "
                        f"outside="
                        f"{replay_result['outside_count']:2d}"
                    )

                results.append(
                    {
                        "k": k,
                        "theta": theta,
                        "N": N,
                        "m": m,
                        "replays": replays,
                        "areas": areas,
                        "replay_results": replay_results,
                        "replay_totals": replay_totals,
                    }
                )

    report_file = create_report(results)

    logger.info(
        "Расчёты завершены. Отчёт: %s",
        report_file
    )


if __name__ == "__main__":
    main()