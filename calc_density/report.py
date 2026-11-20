"""Структура results и запись файлов эксперимента."""

import csv
from dataclasses import astuple
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from uuid import uuid4


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
AREA_KEYS = ('orange', 'blue', 'green')
AREA_LABELS = ('Оранжевая', 'Синяя', 'Зелёная')
METHOD_LABELS = ('Метод 1', 'Метод 2', 'Метод 3')
COEFFICIENT_COLUMNS = ('method_1_D', 'method_2_D', 'method_3_D')
SUMMARY_COLUMNS = (
    'N', 'alpha', 'method', 'mean_error', 'std_error', 'pooled_error', 'clean_expected_error',
    'mean_standard_error', 'expected_error', 'bias_squared', 'variance_coefficient',
    'seed', 'replays', 'N_ref',
)


def _number(value):
    return format(value, '.0f') if float(value).is_integer() else str(value)


def _group_name(k_values):
    return '_'.join(_number(value) for value in k_values)


def _write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _save_figures(directory, suffix, figures):
    for name, figure in figures.items():
        figure.savefig(directory / f'{name}{suffix}.png', dpi=150)


class Report:
    def __init__(self, root=RESULTS_DIR, *, isolated=False):
        self.run_dir = Path(root)
        if isolated:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            self.run_dir /= f'run_{timestamp}_{uuid4().hex[:8]}'
        self.run_dir.mkdir(parents=True, exist_ok=not isolated)

    def group_dir(self, k_values):
        return self.run_dir / f'k_{_group_name(k_values)}'

    def alpha_dir(self, k_values, alpha):
        return self.group_dir(k_values) / f'alpha_{_number(alpha)}'

    def save_scenario(self, experiment, scenario, figures):
        directory = self.alpha_dir(experiment.k_values, scenario.alpha) / f'N_{experiment.N}'
        group = _group_name(experiment.k_values)
        suffix = f'_k_{group}_theta_{_number(experiment.theta)}_N_{experiment.N}'
        directory.mkdir(parents=True, exist_ok=True)
        _save_figures(directory, suffix, figures)
        self._save_replays(directory / f'replays{suffix}.csv', experiment, scenario)
        self._save_areas(directory / f'chi_squared_comparison{suffix}.csv', experiment, scenario)
        _write_csv(directory / f'samples{suffix}.csv',
                   ('replay', 'index', 'component_k', 'clean', 'noisy'), (
            (replay, index, experiment.k_values[component], clean, noisy)
            for replay, (clean_row, noisy_row, labels) in enumerate(
                zip(experiment.clean_samples, scenario.samples, experiment.component_indices), 1
            )
            for index, (clean, noisy, component) in enumerate(
                zip(clean_row.tolist(), noisy_row.tolist(), labels.tolist()), 1
            )
        ))
        (directory / f'report{suffix}.md').write_text(_markdown(experiment, scenario), encoding='utf-8')

    def save_method_comparison(self, k_values, theta, *, alpha, rows, figure):
        directory = self.alpha_dir(k_values, alpha)
        directory.mkdir(parents=True, exist_ok=True)
        stem = f'comparison_three_methods_theta_{_number(theta)}'
        figure.savefig(directory / f'{stem}.png', dpi=150)
        _write_csv(directory / f'{stem}.csv', (
            'N', 'method_1_chi_squared', 'method_2_chi_squared', 'method_3_chi_squared',
            'alpha', 'seed', 'replays', 'N_ref',
        ), (
            (N, *(f'{row.pooled_error:.12g}' for row in methods),
             alpha, methods[0].seed, methods[0].replays, methods[0].N_ref)
            for N, methods in _comparison_by_size(rows)
        ))

    def save_summary(self, k_values, theta, rows):
        directory = self.group_dir(k_values)
        directory.mkdir(parents=True, exist_ok=True)
        _write_csv(directory / f'comparison_theta_{_number(theta)}.csv',
                   ('alpha', 'N', *COEFFICIENT_COLUMNS), _coefficient_rows(rows))

    def save_noise_comparison(self, k_values, theta, figure):
        directory = self.group_dir(k_values)
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / f'noise_comparison_theta_{_number(theta)}.png', dpi=150)

    def save_comparison(self, k_values, theta, *, x_axis, fixed_value, rows, figure):
        directory = (self.alpha_dir(k_values, fixed_value) if x_axis == 'N'
                     else self.group_dir(k_values))
        directory.mkdir(parents=True, exist_ok=True)
        fixed_name = 'alpha' if x_axis == 'N' else 'N'
        stem = f'comparison_by_{x_axis}_{fixed_name}_{_number(fixed_value)}_theta_{_number(theta)}'
        figure.savefig(directory / f'{stem}.png', dpi=150)
        _write_csv(directory / f'{stem}.csv', SUMMARY_COLUMNS, map(astuple, rows))

    def save_all_groups(self, theta, groups):
        _write_csv(self.run_dir / f'comparison_all_groups_theta_{_number(theta)}.csv',
                   ('k_group', *SUMMARY_COLUMNS), (
                       (_group_name(group), *astuple(row))
                       for group, rows in groups.items() for row in rows
                   ))

    def save_group_comparison(self, theta, *, fixed_axis, fixed_value, groups, figure):
        stem = f'comparison_all_groups_theta_{_number(theta)}_{fixed_axis}_{_number(fixed_value)}'
        figure.savefig(self.run_dir / f'{stem}.png', dpi=150)
        _write_csv(self.run_dir / f'{stem}.csv',
                   ('k_group', 'N', 'alpha', *COEFFICIENT_COLUMNS),
                   _group_coefficient_rows(groups))

    def save_validation(self, control_columns, control_rows, notes):
        _write_csv(self.run_dir / 'fixed_boundaries_control.csv', control_columns, control_rows)
        (self.run_dir / 'validation.md').write_text(notes, encoding='utf-8')

    @staticmethod
    def _save_replays(path, experiment, scenario):
        columns = (
            'replay',
            *(f'{prefix}{area}_count' for prefix in ('', 'method2_', 'method3_') for area in AREA_KEYS),
            'total',
        )
        mixed = len(experiment.k_values) > 1
        quotas = sorted(
            (f'selected_from_k_{_number(k)}', count)
            for k, count in zip(experiment.k_values, experiment.quotas)
        ) if mixed else []
        _write_csv(path, (*columns, *(key for key, _ in quotas)), (
            (replay, *(count for counts in methods for count in counts), experiment.N,
             *(count for _, count in quotas))
            for replay, methods in enumerate(zip(*(result.counts for result in scenario.methods)), 1)
        ))

    @staticmethod
    def _save_areas(path, experiment, scenario):
        total = experiment.N * len(experiment.clean_samples)
        _write_csv(path, (
            'method', 'area', 'theoretical_probability_p', 'empirical_probability_p_hat',
            'observed_count', 'total_count', 'chi_squared_contribution',
            'count_difference_observed_minus_theoretical', 'relative_deviation_percent',
            'chi_squared_total',
        ), (
            (method, label, f'{area.p:.12g}', f'{area.p_hat:.12g}', area.observed, total,
             f'{area.pooled_contribution:.12g}', ceil(area.difference),
             f'{area.relative_deviation_percent:.12g}', f'{result.pooled_error:.12g}')
            for method, result in zip(METHOD_LABELS, scenario.methods)
            for label, area in zip(AREA_LABELS, result.areas)
        ))


def _coefficient_rows(rows):
    by_case = {}
    for row in rows:
        by_case.setdefault((row.alpha, row.N), {})[row.method] = row.pooled_error
    for (alpha, N), coefficients in sorted(by_case.items()):
        yield alpha, N, *(coefficients[method] for method in (1, 2, 3))


def _group_coefficient_rows(groups):
    for group, rows in groups.items():
        for alpha, N, *coefficients in _coefficient_rows(rows):
            yield _group_name(group), N, alpha, *coefficients


def _comparison_by_size(rows):
    data = {}
    for row in rows:
        data.setdefault(row.N, []).append(row)
    for N, methods in sorted(data.items()):
        yield N, sorted(methods, key=lambda row: row.method)


def _markdown(experiment, scenario):
    k_values = experiment.k_values
    k = _number(k_values[0]) if len(k_values) == 1 else f'[{", ".join(map(_number, k_values))}]'
    replays = len(experiment.clean_samples)
    total = experiment.N * replays
    lines = [
        '# Результаты эксперимента', '',
        f'`k = {k}`, `θ = {experiment.theta:g}`, `N = {experiment.N}`, `replays = {replays}`',
        '',
        f'`α = {scenario.alpha:g}`, `koef = {experiment.koef:g}`, `seed = {experiment.seed}`', '',
        f'Границы зафиксированы при `N_ref = {experiment.N_ref}` для всей серии N и α.', '',
    ]
    if len(k_values) > 1:
        quotas = ', '.join(
            f'k={value:g}: {count} элементов'
            for value, count in zip(k_values, experiment.quotas)
        )
        lines += [f'Квоты смешанной выборки: {quotas}.', '']
    lines += [
        '## 1. Результаты χ²', '',
        'Формула: `χ² = Σ ((p_j - p̂_j) / p_j)²`. '
        'Здесь p_j — вероятность по чистой модели, p̂_j — доля по всем повторениям.',
        '',
        '| Метод | Область | Теоретическая вероятность p_j | Экспериментальная вероятность p̂_j | Вклад в χ² |',
        '|---|---|---:|---:|---:|',
    ]
    for method, result in zip(METHOD_LABELS, scenario.methods):
        for label, area in zip(AREA_LABELS, result.areas):
            lines.append(f'| {method} | {label} | {area.p:.8f} | {area.p_hat:.8f} | '
                         f'{area.pooled_contribution:.10f} |')
        lines.append(f'| **{method}: итог χ²** | — | — | — | **{result.pooled_error:.10f}** |')
    lines += [
        '', '## 2. Сравнение теоретического и фактического количества точек', '',
        f'Теоретическое количество рассчитано для всех {replays} повторений. '
        'Разница в точках округляется вверх. Процент рассчитан до округления.',
        '',
        '| Метод | Область | Теоретическое количество точек | Фактическое количество точек | '
        'Разница (факт − теория), точек | Отклонение от теории, % |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for method, result in zip(METHOD_LABELS, scenario.methods):
        for label, area in zip(AREA_LABELS, result.areas):
            lines.append(f'| {method} | {label} | {area.expected:.4f} | {area.observed} | '
                         f'{ceil(area.difference):+d} | {area.relative_deviation_percent:+.2f}% |')
        lines.append(f'| **{method}: всего** | — | **{total:.4f}** | **{total}** | **+0** | **+0.00%** |')
    lines += [
        '', '## 3. Границы и ошибка по повторениям', '',
        '| Метод | a | b | Средняя D | СКО D | Стандартная ошибка среднего | E[D] при данном α | E[D] при α=0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for method, part, result in zip(METHOD_LABELS, experiment.partitions, scenario.methods):
        a, b = part.boundaries
        se = result.mean_standard_error
        se_text = f'{se:.12g}' if se is not None else '—'
        lines.append(f'| {method} | {a:.12g} | {b:.12g} | {result.mean_error:.12g} | '
                     f'{result.std_error:.12g} | {se_text} | {result.expected_error:.12g} | '
                     f'{part.clean_expected_error:.12g} |')
    return '\n'.join(lines) + '\n'
