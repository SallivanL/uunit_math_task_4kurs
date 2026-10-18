"""Построение графиков. Модуль только возвращает figure и не сохраняет файлы."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from calc import normalize_distribution_config


AREA_COLORS = {"orange": "#E99A37", "blue": "#438BD0", "green": "#42966D"}
AREA_LABELS = {"orange": "Оранжевая", "blue": "Синяя", "green": "Зелёная"}


def _title(distribution, N, suffix=""):
    config = normalize_distribution_config(distribution)
    text = f"{config['label']}: {config['params_text']}, N={N}"
    return f"{text} — {suffix}" if suffix else text


def plot_three_densities(x, f, f_min, f_max, areas, distribution, N):
    """Строит исходную плотность и плотности минимума/максимума."""
    figure, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x, f_max, linewidth=2, label=r"$f_{max}(x)$ — максимум")
    ax.plot(x, f_min, linewidth=2, label=r"$f_{min}(x)$ — минимум")
    ax.plot(x, f, linewidth=2, label=r"$f(x)$ — исходная плотность")

    orange = areas["orange"]
    blue = areas["blue"]
    green = areas["green"]
    ax.fill_between(x[orange["left"]:orange["right"] + 1],
                    f_min[orange["left"]:orange["right"] + 1],
                    color=AREA_COLORS["orange"], alpha=0.25, label="Левая область")
    ax.fill_between(x[blue["left"]:blue["right"] + 1],
                    f[blue["left"]:blue["right"] + 1],
                    color=AREA_COLORS["blue"], alpha=0.25, label="Центральная область")
    ax.fill_between(x[green["left"]:green["right"] + 1],
                    f_max[green["left"]:green["right"] + 1],
                    color=AREA_COLORS["green"], alpha=0.25, label="Правая область")
    ax.set_title(_title(distribution, N, "плотности исходного закона, минимума и максимума"), fontsize=13)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Плотность", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    figure.tight_layout()
    return figure


def plot_cdf(x, cdf_area_probabilities, distribution, N):
    """Строит CDF выбранного распределения и отмечает границы метода 1."""
    figure, ax = plt.subplots(figsize=(11, 7))
    x1 = cdf_area_probabilities["orange_boundary"]
    x2 = cdf_area_probabilities["green_boundary"]
    y1 = cdf_area_probabilities["cdf_orange"]
    y2 = cdf_area_probabilities["cdf_green"]
    ax.plot(x, cdf_area_probabilities["cdf_values"], linewidth=2, label=r"$F(x)$ — функция распределения")
    ax.scatter([x1, x2], [y1, y2], s=55, zorder=5, label="Границы областей")
    for xb, yb in ((x1, y1), (x2, y2)):
        ax.vlines(xb, 0, yb, linestyles="--", linewidth=1.3)
        ax.hlines(yb, min(0.0, float(np.min(x))), xb, linestyles="--", linewidth=1.3)
    ax.annotate(f"F(x₁)={y1:.4f}\nx₁={x1:.4f}", (x1, y1), xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.annotate(f"F(x₂)={y2:.4f}\nx₂={x2:.4f}", (x2, y2), xytext=(8, -32), textcoords="offset points", fontsize=10)
    ax.set_title(_title(distribution, N, "функция распределения F(x)"), fontsize=13)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("F(x)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    figure.tight_layout()
    return figure


def plot_replay_histogram(average_replay_ratios, distribution, N, replays):
    """Строит итоговую гистограмму средних долей трёх областей."""
    figure, ax = plt.subplots(figsize=(9, 7))
    keys = ("orange_ratio", "blue_ratio", "green_ratio")
    values = [average_replay_ratios[key] for key in keys]
    labels = ["Область 1", "Область 2", "Область 3"]
    colors = [AREA_COLORS["orange"], AREA_COLORS["blue"], AREA_COLORS["green"]]
    bars = ax.bar(labels, values, width=0.6, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Средняя доля точек n / N")
    ax.set_xlabel("Область")
    ax.set_title(_title(distribution, N, f"средние доли по {replays} replay"), fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def plot_replays_histogram(replay_results, N, distribution):
    """Строит три столбца на каждый replay для основного метода."""
    if N <= 0 or not replay_results:
        raise ValueError("N должно быть > 0, а replay_results не должен быть пустым.")
    replay_numbers = [int(r["replay"]) for r in replay_results]
    counts_by_area = {area: [int(r[f"{area}_count"]) for r in replay_results]
                      for area in ("orange", "blue", "green")}
    for number, counts in zip(replay_numbers, zip(*(counts_by_area[a] for a in ("orange", "blue", "green")))):
        if sum(counts) != N:
            raise ValueError(f"Сумма точек replay {number} не равна N.")
    x_pos = np.arange(len(replay_results))
    width = 0.24
    figure_width = max(11, len(replay_results) * 1.1)
    figure, ax = plt.subplots(figsize=(figure_width, 7))
    offsets = {"orange": -width, "blue": 0, "green": width}
    for area in ("orange", "blue", "green"):
        counts = counts_by_area[area]
        ratios = [count / N for count in counts]
        bars = ax.bar(x_pos + offsets[area], ratios, width=width,
                      label=AREA_LABELS[area], color=AREA_COLORS[area])
        for bar, count, ratio in zip(bars, counts, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, ratio / 2 if ratio > 0 else 0.012,
                    str(count), ha="center", va="center" if ratio > 0 else "bottom", fontsize=8,
                    color="white" if ratio >= 0.12 else "black", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(number) for number in replay_numbers])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Номер replay")
    ax.set_ylabel("Доля точек n / N")
    ax.set_title(_title(distribution, N, "распределение точек по областям для каждого replay"), fontsize=13)
    ax.legend(title="Область")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_cdf_histogram(cdf_area_probabilities, distribution, N):
    """Строит гистограмму теоретических вероятностей метода 1."""
    return _plot_probability_histogram(
        [cdf_area_probabilities["orange_probability"], cdf_area_probabilities["blue_probability"],
         cdf_area_probabilities["green_probability"]],
        distribution, N, "Теоретические вероятности (метод 1)")


def plot_equal_cdf(x, equal_cdf_areas, distribution, N):
    """Строит CDF и границы метода 2 (F(x)=1/3 и F(x)=2/3)."""
    figure, ax = plt.subplots(figsize=(11, 7))
    x1, x2 = equal_cdf_areas["orange_boundary"], equal_cdf_areas["green_boundary"]
    y1, y2 = equal_cdf_areas["cdf_orange"], equal_cdf_areas["cdf_green"]
    cdf_values = equal_cdf_areas.get("cdf_values")
    if cdf_values is None:
        raise ValueError("В equal_cdf_areas отсутствуют cdf_values.")
    ax.plot(x, cdf_values, linewidth=2, label=r"$F(x)$ — функция распределения")
    ax.scatter([x1, x2], [y1, y2], s=55, zorder=5, label="Границы равных областей")
    for xb, yb in ((x1, y1), (x2, y2)):
        ax.vlines(xb, 0, yb, linestyles="--", linewidth=1.3)
        ax.hlines(yb, min(0.0, float(np.min(x))), xb, linestyles="--", linewidth=1.3)
    ax.annotate(f"F(x₁)={y1:.4f}\nx₁={x1:.4f}", (x1, y1), xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.annotate(f"F(x₂)={y2:.4f}\nx₂={x2:.4f}", (x2, y2), xytext=(8, -32), textcoords="offset points", fontsize=10)
    ax.set_title(_title(distribution, N, "метод 2: три равные области по F(x)"), fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10)
    figure.tight_layout()
    return figure


def plot_equal_cdf_histogram(equal_cdf_areas, distribution, N):
    """Строит гистограмму теоретических вероятностей метода 2."""
    return _plot_probability_histogram(
        [equal_cdf_areas["orange_probability"], equal_cdf_areas["blue_probability"],
         equal_cdf_areas["green_probability"]],
        distribution, N, "Теоретические вероятности (метод 2)")


def plot_method2_replays_histogram(replay_results, N, distribution):
    """Строит один рисунок с гистограммами всех replay для метода 2."""
    if N <= 0 or not replay_results:
        raise ValueError("N должно быть > 0, а replay_results не должен быть пустым.")
    mapped = []
    for r in replay_results:
        mapped.append({
            "replay": r["replay"],
            "orange_count": r["method2_orange_count"],
            "blue_count": r["method2_blue_count"],
            "green_count": r["method2_green_count"],
            "total": r["total"],
        })
    return plot_replays_histogram(mapped, N, distribution)


def _plot_probability_histogram(values, distribution, N, suffix):
    figure, ax = plt.subplots(figsize=(9, 7))
    labels = ["Область 1", "Область 2", "Область 3"]
    colors = [AREA_COLORS["orange"], AREA_COLORS["blue"], AREA_COLORS["green"]]
    bars = ax.bar(labels, values, width=0.6, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Теоретическая вероятность")
    ax.set_xlabel("Область")
    ax.set_title(_title(distribution, N, suffix), fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure
