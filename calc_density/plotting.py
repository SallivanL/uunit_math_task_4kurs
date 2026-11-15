"""Графики из готовых результатов эксперимента."""

from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
import numpy as np


COLORS = ('#d98b31', '#438bd0', '#42966d')
AREA_NAMES = ('Оранжевая', 'Синяя', 'Зелёная')


def _caption(experiment, alpha=None):
    k = ', '.join(f'{value:g}' for value in experiment.k_values)
    parameters = f'k={k}, θ={experiment.theta:g}, N={experiment.N}'
    noise = 'чистая модель' if alpha is None else f'α={alpha:g}'
    return f'{parameters}, {noise}\nГраницы при N_ref={experiment.N_ref}'


def _axes(title, xlabel, ylabel):
    figure = Figure(figsize=(10, 5.8), layout='constrained')
    ax = figure.subplots()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis='y', alpha=0.24)
    ax.set_axisbelow(True)
    return figure, ax


def _mark_boundaries(ax, boundaries):
    for boundary in boundaries:
        ax.axvline(boundary, color='#444444', linestyle='--', linewidth=1)
        ax.annotate(f'{boundary:.5g}', xy=(boundary, 0.98), xycoords=('data', 'axes fraction'),
                    xytext=(4, 0), textcoords='offset points', va='top', fontsize=9)


def plot_densities(experiment, method):
    part = experiment.partitions[method - 1]
    curves = experiment.curves
    title = {
        1: f'Метод 1 · площади хвостов = {experiment.koef:.0%}',
        3: 'Метод 3 · равные интервалы расчётного диапазона',
    }[method]
    figure, ax = _axes(f'{title}\n{_caption(experiment)}', 'x', 'Плотность')
    x = curves.x
    ax.plot(x, curves.maximum_density, label=r'$f_{\max}(x)$ — максимум')
    ax.plot(x, curves.minimum_density, label=r'$f_{\min}(x)$ — минимум')
    ax.plot(x, curves.density, label=r'$f(x)$ — исходное распределение')
    a, b = part.boundaries
    masks = (x <= a, (x >= a) & (x <= b), x >= b)
    layers = (curves.minimum_density, curves.density, curves.maximum_density)
    if method == 3:
        layers = (curves.density,) * 3
    for mask, density, color in zip(masks, layers, COLORS):
        ax.fill_between(x, 0, density, where=mask, color=color, alpha=0.24)
    _mark_boundaries(ax, part.boundaries)
    ax.set(xlim=(0, part.x_max or x[-1]), ylim=(0, None))
    ax.legend(loc='center right')
    return figure


def plot_cdf(experiment, method):
    part = experiment.partitions[method - 1]
    curves = experiment.curves
    figure, ax = _axes(f'Метод {method} · границы на CDF\n{_caption(experiment)}', 'x', 'F(x)')
    ax.plot(np.r_[0, curves.x], np.r_[0, curves.cdf])
    for boundary, level in zip(part.boundaries, part.cdf_levels):
        ax.plot([0, boundary, boundary], [level, level, 0], '--', color='#888888', linewidth=1)
        ax.scatter(boundary, level, color='#333333', zorder=3)
    _mark_boundaries(ax, part.boundaries)
    ax.set(xlim=(0, part.x_max or curves.x[-1]), ylim=(0, 1.08))
    return figure


def plot_probabilities(experiment, method, scenario=None):
    part = experiment.partitions[method - 1]
    caption = _caption(experiment, None if scenario is None else scenario.alpha)
    figure, ax = _axes(f'Метод {method} · доли наблюдений\n{caption}', 'Область', 'Доля')
    positions = np.arange(3)
    if scenario is None:
        ax.bar(positions, part.probabilities, color=COLORS, label='Чистая теория')
    else:
        result = scenario.methods[method - 1]
        shares = [area.p_hat for area in result.areas]
        ax.bar(positions - 0.18, part.probabilities, 0.36, color='#a0a9b4', label='Чистая теория')
        ax.bar(positions + 0.18, shares, 0.36, color=COLORS, label='Фактические доли')
    ax.set_xticks(positions, AREA_NAMES)
    ax.set_ylim(0, 1.06)
    ax.legend()
    return figure


def plot_replay_histogram(experiment, scenario):
    """Средние фактические доли трёх областей первого метода."""
    values = [area.p_hat for area in scenario.methods[0].areas]
    figure, ax = _axes(
        f'Среднее распределение точек по областям\n{_caption(experiment, scenario.alpha)}, '
        f'повторений={len(experiment.clean_samples)}',
        'Область', 'Средняя доля точек n / N',
    )
    bars = ax.bar(('Область 1', 'Область 2', 'Область 3'), values, color=COLORS, width=0.6)
    ax.bar_label(bars, fmt='%.4f', padding=3)
    ax.set_ylim(0, 1.08)
    return figure


def plot_replays(experiment, scenario, method):
    result = scenario.methods[method - 1]
    counts = result.counts
    caption = _caption(experiment, scenario.alpha)
    figure, ax = _axes(f'Метод {method} · повторения\n{caption}',
                       'Номер повторения', 'Доля')
    positions = np.arange(1, len(counts) + 1)
    for offset, color, name, group, fractions in zip(
        (-0.24, 0, 0.24), COLORS, AREA_NAMES, counts.T, result.fractions.T
    ):
        if len(counts) > 20:
            # Все столбцы одной области рисуем одной коллекцией.
            centers = positions + offset
            x = centers[:, None] + np.array([-0.12, -0.12, 0.12, 0.12])
            y = fractions[:, None] * np.array([0, 1, 1, 0])
            ax.add_collection(PolyCollection(np.stack((x, y), axis=-1),
                                             facecolors=color, label=name))
            continue
        bars = ax.bar(positions + offset, fractions, 0.24, color=color, label=name)
        if len(counts) <= 10:
            ax.bar_label(bars, labels=[str(count) for count in group], fontsize=8, padding=2)
    tick_step = max(1, (len(counts) + 19) // 20)
    ax.set_xticks(positions[::tick_step])
    ax.set_xlim(0.5, len(counts) + 0.5)
    ax.set_ylim(0, 1.08)
    ax.legend()
    return figure


def plot_comparison_three_methods(rows, *, k_values, theta, alpha):
    """Объединённая ошибка по N на одном графике для всех трёх методов."""
    k = ', '.join(f'{value:g}' for value in k_values)
    figure, ax = _axes(
        f'Объединённая ошибка трёх методов\nk={k}, θ={theta:g}, α={alpha:g}\n'
        f'Границы при N_ref={rows[0].N_ref}',
        'Размер выборки N', r'$\chi^2$',
    )
    for method, marker in enumerate(('o', 's', '^'), 1):
        data = [row for row in rows if row.method == method]
        ax.plot([row.N for row in data], [row.pooled_error for row in data],
                marker=marker, label=f'Метод {method}')
    ax.set_xticks(sorted({row.N for row in rows}))
    ax.set_ylim(bottom=0)
    ax.legend()
    return figure


def plot_comparison(rows, *, x_axis, fixed_value, k_values, theta):
    figure = Figure(figsize=(13, 5), layout='constrained')
    axes = figure.subplots(1, 3, sharey=False)
    k = ', '.join(f'{value:g}' for value in k_values)
    fixed_label = 'α' if x_axis == 'N' else 'N'
    xlabel = 'Размер выборки N' if x_axis == 'N' else 'Уровень шума α'
    figure.suptitle(f'Средняя ошибка по повторениям\n'
                    f'k={k}, θ={theta:g}, {fixed_label}={fixed_value:g} · '
                    f'границы при N_ref={rows[0].N_ref}')
    for method, ax in enumerate(axes, 1):
        data = [row for row in rows if row.method == method]
        x = [getattr(row, x_axis) for row in data]
        ax.plot(x, [row.expected_error for row in data], '--', color='#555555', label='E[D]')
        means = [row.mean_error for row in data]
        if all(row.mean_standard_error is not None for row in data):
            ax.errorbar(x, means, yerr=[1.96 * row.mean_standard_error for row in data],
                        fmt='o', capsize=3, label='Среднее D и 95% интервал')
        else:
            ax.plot(x, means, 'o', label='Среднее D')
        ax.legend()
        ax.set(title=f'Метод {method}', xlabel=xlabel, ylabel='D', ylim=(0, None))
        ax.set_xticks(x)
        ax.grid(axis='y', alpha=0.24)
    return figure


def plot_noise_comparison(rows, *, k_values, theta):
    """Ожидаемый коэффициент в зависимости от α для всех размеров N."""
    figure = Figure(figsize=(13, 5), layout='constrained')
    axes = figure.subplots(1, 3)
    k = ', '.join(f'{value:g}' for value in k_values)
    figure.suptitle(f'Влияние шума на ожидаемую ошибку\nk={k}, θ={theta:g} · '
                    f'границы при N_ref={rows[0].N_ref}')
    sizes = sorted({row.N for row in rows})
    for method, ax in enumerate(axes, 1):
        for N in sizes:
            series = sorted((row for row in rows if row.method == method and row.N == N),
                            key=lambda row: row.alpha)
            ax.plot([row.alpha for row in series], [row.expected_error for row in series],
                    'o-', label=f'N={N}')
        ax.set(title=f'Метод {method}', xlabel='Уровень шума α', ylabel='E[D]', ylim=(0, None))
        ax.set_xticks(sorted({row.alpha for row in rows}))
        ax.grid(axis='y', alpha=0.24)
    axes[0].legend()
    return figure


def plot_group_comparison(groups, *, theta, x_axis, fixed_value):
    """Сравнивает D по объединённым повторам для групп k."""
    figure = Figure(figsize=(13, 5), layout='constrained')
    axes = figure.subplots(1, 3)
    fixed_label = 'N' if x_axis == 'alpha' else 'α'
    xlabel = 'Уровень шума α' if x_axis == 'alpha' else 'Размер выборки N'
    figure.suptitle(f'Коэффициент D по объединённой выборке\n'
                    f'θ={theta:g}, {fixed_label}={fixed_value:g}')
    for method, ax in enumerate(axes, 1):
        ticks = set()
        for group, rows in groups.items():
            series = sorted((row for row in rows if row.method == method),
                            key=lambda row: getattr(row, x_axis))
            x = [getattr(row, x_axis) for row in series]
            ticks.update(x)
            label = ', '.join(f'{value:g}' for value in group)
            ax.plot(x, [row.pooled_error for row in series], 'o-', label=f'k={label}')
        ax.set(title=f'Метод {method}', xlabel=xlabel, ylabel='D', ylim=(0, None))
        ax.set_xticks(sorted(ticks))
        ax.grid(axis='y', alpha=0.24)
    axes[0].legend()
    return figure
