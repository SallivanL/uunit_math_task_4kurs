"""Проверка зависимости D от N для четырёх групп на 1000 повторениях."""

from collections import defaultdict

from calc import run_experiments, summarize
from report import RESULTS_DIR, Report


SEED = 20261115
REPLAYS = 1000
REFERENCE_SIZE = 20


def run_study(results_root):
    cases = run_experiments(
        k_values=(), massive_k=((1,), (8,), (1, 8), (2, 4)), theta_values=(1,),
        N_values=(20, 100, 500, 1000), alpha_values=(0, 0.2, 0.5, 1, 2),
        koef=0.7, replays=REPLAYS, seed=SEED, N_ref=REFERENCE_SIZE, with_curves=False,
    )
    summaries = defaultdict(list)
    expected_by_N = defaultdict(list)
    largest_z = 0.0
    for case in cases:
        rows = summarize(case)
        summaries[case.k_values].extend(rows)
        for row in rows:
            key = (case.k_values, row.alpha, row.method)
            expected_by_N[key].append((row.N, row.expected_error, row.mean_error))
            deviation = abs(row.mean_error - row.expected_error)
            if deviation > 5 * row.mean_standard_error:
                raise AssertionError(f'Среднее D вышло за 5 стандартных ошибок: {key}, N={row.N}')
            if row.mean_standard_error:
                largest_z = max(largest_z, deviation / row.mean_standard_error)

    comparisons = []
    sample_decreases = 0
    for (group, alpha, method), series in expected_by_N.items():
        for (N1, expected1, mean1), (N2, expected2, mean2) in zip(series, series[1:]):
            if not expected2 < expected1:
                raise AssertionError(f'E[D] не убывает: {group}, α={alpha}, метод {method}')
            sample_decreases += mean2 < mean1
            k = '_'.join(f'{value:g}' for value in group)
            comparisons.append((k, alpha, method, N1, N2, expected1, expected2, mean1, mean2))

    report = Report(results_root, isolated=True)
    for group, rows in summaries.items():
        report.save_summary(group, 1, rows)
    report.save_all_groups(1, summaries)
    notes = [
        '# Проверка зависимости коэффициента D от размера выборки и шума',
        f'Границы зафиксированы при N_ref={REFERENCE_SIZE}. Для каждой комбинации '
        f'выполнено {REPLAYS} повторений с seed={SEED}.',
        f'E[D] строго уменьшается с N во всех {len(expected_by_N)} сериях. '
        f'Среднее по повторениям снижается в {sample_decreases} из {len(comparisons)} '
        'пар соседних размеров.',
        f'Максимальное отклонение среднего от E[D] составило {largest_z:.2f} '
        'стандартной ошибки среднего.',
    ]
    report.save_validation(
        ('k', 'alpha', 'method', 'N1', 'N2', 'expected_D_N1', 'expected_D_N2',
         'mean_D_N1', 'mean_D_N2'),
        comparisons, '\n\n'.join(notes) + '\n',
    )
    return report.run_dir


if __name__ == '__main__':
    print(run_study(RESULTS_DIR / 'validation'))
