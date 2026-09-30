import os
import shutil

import numpy as np

from calc import (
    calculate_densities,
    calculate_areas,
    run_replays,
    calculate_replay_totals,
    calculate_area_statistics,
)
from plotting import plot_three_densities
from report import save_experiment


def main():
    m = 0.8
    replays = 40

    k_values = [
        1,
        8,
    ]

    theta_values = [1]
    N_values = [30]

    random_seed = None

    if os.path.exists("results"):
        shutil.rmtree("results")

    rng = np.random.default_rng(random_seed)

    for k in k_values:
        for theta in theta_values:
            for N in N_values:
                x, f, f_min, f_max = calculate_densities(
                    k=k,
                    theta=theta,
                    N=N,
                )

                areas = calculate_areas(
                    x=x,
                    f=f,
                    f_min=f_min,
                    f_max=f_max,
                    m=m,
                )

                figure = plot_three_densities(
                    x=x,
                    f=f,
                    f_min=f_min,
                    f_max=f_max,
                    areas=areas,
                    k=k,
                    theta=theta,
                    N=N,
                )

                replay_results = run_replays(
                    k=k,
                    theta=theta,
                    N=N,
                    replays=replays,
                    x=x,
                    areas=areas,
                    rng=rng,
                )

                replay_totals = calculate_replay_totals(
                    replay_results
                )

                area_statistics = calculate_area_statistics(
                    k=k,
                    theta=theta,
                    N=N,
                    x=x,
                    areas=areas,
                    replay_results=replay_results,
                )

                save_experiment(
                    k=k,
                    theta=theta,
                    N=N,
                    m=m,
                    replays=replays,
                    x=x,
                    areas=areas,
                    figure=figure,
                    replay_results=replay_results,
                    replay_totals=replay_totals,
                    area_statistics=area_statistics,
                )


if __name__ == "__main__":
    main()