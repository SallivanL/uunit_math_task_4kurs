"""Расчёт → построение рисунков → сохранение результатов."""

from collections import defaultdict
from itertools import groupby

from calc import run_experiments, summarize
from plotting import (
    plot_cdf,
    plot_comparison,
    plot_comparison_three_methods,
    plot_densities,
    plot_group_comparison,
    plot_noise_comparison,
    plot_probabilities,
    plot_replay_histogram,
    plot_replays,
)
from report import Report


def main(
    *, k_values=(1, 8), massive_k=((1, 6)), theta_values=(1,),
    N_values=(20, 100, 400), N_ref=20, alpha_values=(0, 0.2, 0.5, 1, 2),
    koef=0.7, replays=100, seed=20261115, results_root=None,
):
    """massive_k=[1, 8, 15] задаёт смесь, [[1, 8], [2, 4]] — две независимые смеси."""
    experiments = run_experiments(
        k_values=k_values, massive_k=massive_k, theta_values=theta_values, N_values=N_values,
        alpha_values=alpha_values, koef=koef, replays=replays, seed=seed, N_ref=N_ref,
    )
    report = None
    summaries = defaultdict(dict)
    for (group, theta), cases in groupby(experiments, key=lambda case: (case.k_values, case.theta)):
        if report is None:
            report = Report() if results_root is None else Report(results_root)
        rows = []
        for case in cases:
            alphas = [scenario.alpha for scenario in case.noise_results]
            theory_figures = {
                'initial_graphs': plot_densities(case, 1),
                'method_3_distribution': plot_densities(case, 3),
            }
            for method in range(1, 4):
                theory_figures[f'method_{method}_cdf'] = plot_cdf(case, method)
                theory_figures[f'method_{method}_theoretical_probabilities'] = (
                    plot_probabilities(case, method)
                )
            for scenario in case.noise_results:
                figures = {'initial_histogram': plot_replay_histogram(case, scenario)}
                for method in range(1, 4):
                    figures[f'method_{method}_replay_histograms'] = (
                        plot_replays(case, scenario, method)
                    )
                    if len(alphas) > 1:
                        figures[f'method_{method}_observed_probabilities'] = (
                            plot_probabilities(case, method, scenario)
                        )
                report.save_scenario(case, scenario, theory_figures | figures)
            rows.extend(summarize(case))

        rows.sort(key=lambda row: (row.N, row.alpha, row.method))
        for alpha in sorted(alphas):
            selected = [row for row in rows if row.alpha == alpha]
            figure = plot_comparison_three_methods(selected, k_values=group, theta=theta, alpha=alpha)
            report.save_method_comparison(group, theta, alpha=alpha, rows=selected, figure=figure)
        report.save_summary(group, theta, rows)
        summaries[theta][group] = rows
        axes = [('N', 'alpha')]
        if len(alphas) > 1:
            noise_figure = plot_noise_comparison(rows, k_values=group, theta=theta)
            report.save_noise_comparison(group, theta, noise_figure)
            axes.append(('alpha', 'N'))
        for x_axis, fixed_axis in axes:
            for fixed_value in sorted({getattr(row, fixed_axis) for row in rows}):
                selected = [row for row in rows if getattr(row, fixed_axis) == fixed_value]
                figure = plot_comparison(
                    selected, x_axis=x_axis, fixed_value=fixed_value, k_values=group, theta=theta,
                )
                report.save_comparison(
                    group, theta, x_axis=x_axis, fixed_value=fixed_value, rows=selected,
                    figure=figure,
                )
    for theta, groups in summaries.items():
        report.save_all_groups(theta, groups)
        save_group_comparisons(report, theta, groups)
    return report.run_dir


def save_group_comparisons(report, theta, groups):
    for x_axis, fixed_axis in (('alpha', 'N'), ('N', 'alpha')):
        values = sorted({getattr(row, fixed_axis) for rows in groups.values() for row in rows})
        for value in values:
            selected = {
                group: [row for row in rows if getattr(row, fixed_axis) == value]
                for group, rows in groups.items()
            }
            figure = plot_group_comparison(selected, theta=theta, x_axis=x_axis, fixed_value=value)
            report.save_group_comparison(theta, fixed_axis=fixed_axis, fixed_value=value,
                                         groups=selected, figure=figure)


if __name__ == '__main__':
    main()
