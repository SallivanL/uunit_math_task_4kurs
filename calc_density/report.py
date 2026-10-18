"""Единый модуль структуры результатов и сохранения PNG, Markdown и CSV."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from calc import normalize_distribution_config


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def clear_results() -> None:
    """Удаляет результаты предыдущего полного запуска (вызывается из main один раз)."""
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)


def _safe_token(value: Any) -> str:
    """Преобразует часть параметра в безопасный компонент имени папки."""
    text = f"{float(value):g}" if isinstance(value, (int, float)) else str(value)
    text = text.replace("-", "m").replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return text.strip("_") or "value"


def get_experiment_dir(distribution: dict[str, Any], N: int) -> Path:
    """Создаёт отдельную директорию закона, его параметров и размера выборки."""
    config = normalize_distribution_config(distribution)
    param_tokens = [f"{key}_{_safe_token(value)}" for key, value in config["params"].items()]
    params_dirname = "params_" + "_".join(param_tokens)
    directory = RESULTS_DIR / f"{config['id']}_{config['key']}" / params_dirname / f"N_{int(N)}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_experiment(
    distribution: dict[str, Any],
    N: int,
    m: float,
    replays: int,
    x,
    areas: dict[str, Any],
    figure,
    cdf_figure,
    histogram,
    replay_histogram,
    method2_replay_histogram,
    cdf_histogram,
    equal_cdf_figure,
    equal_cdf_histogram,
    equal_cdf_areas: dict[str, Any],
    method2_replay_totals: dict[str, Any],
    replay_results: list[dict[str, Any]],
    replay_ratios: list[dict[str, Any]],
    average_replay_ratios: dict[str, Any],
    replay_totals: dict[str, Any],
    area_statistics: dict[str, Any],
    cdf_area_probabilities: dict[str, Any],
) -> Path:
    """Сохраняет все артефакты одного эксперимента; возвращает папку эксперимента."""
    config = normalize_distribution_config(distribution)
    output_dir = get_experiment_dir(config, N)

    figures = {
        "density_min_max.png": figure,
        "cdf_method_1.png": cdf_figure,
        "average_replay_shares.png": histogram,
        "all_replays_method_1.png": replay_histogram,
        "all_replays_method_2.png": method2_replay_histogram,
        "probabilities_method_1.png": cdf_histogram,
        "cdf_method_2.png": equal_cdf_figure,
        "probabilities_method_2.png": equal_cdf_histogram,
    }
    for filename, figure_obj in figures.items():
        if figure_obj is None:
            continue
        try:
            figure_obj.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        finally:
            plt.close(figure_obj)

    _save_replays_csv(
        output_dir=output_dir,
        distribution=config,
        N=N,
        replay_results=replay_results,
    )
    _save_markdown_report(
        output_dir=output_dir,
        distribution=config,
        N=N,
        m=m,
        replays=replays,
        x=x,
        areas=areas,
        replay_results=replay_results,
        replay_ratios=replay_ratios,
        average_replay_ratios=average_replay_ratios,
        replay_totals=replay_totals,
        area_statistics=area_statistics,
        cdf_area_probabilities=cdf_area_probabilities,
        equal_cdf_areas=equal_cdf_areas,
        method2_replay_totals=method2_replay_totals,
    )
    return output_dir


def _save_replays_csv(output_dir: Path, distribution: dict[str, Any], N: int,
                      replay_results: list[dict[str, Any]]) -> None:
    """Сохраняет фактические количества точек каждого replay в CSV."""
    config = normalize_distribution_config(distribution)
    filename = output_dir / "replays.csv"
    fieldnames = [
        "distribution_id", "distribution_key", "distribution_name", "distribution_params",
        "N", "replay", "orange_count", "blue_count", "green_count",
        "method2_orange_count", "method2_blue_count", "method2_green_count", "total",
    ]
    with filename.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in replay_results:
            writer.writerow({
                "distribution_id": config["id"],
                "distribution_key": config["key"],
                "distribution_name": config["label"],
                "distribution_params": config["params_text"],
                "N": int(N),
                "replay": result["replay"],
                "orange_count": result["orange_count"],
                "blue_count": result["blue_count"],
                "green_count": result["green_count"],
                "method2_orange_count": result.get("method2_orange_count", ""),
                "method2_blue_count": result.get("method2_blue_count", ""),
                "method2_green_count": result.get("method2_green_count", ""),
                "total": result["total"],
            })


def _save_markdown_report(
    output_dir: Path,
    distribution: dict[str, Any],
    N: int,
    m: float,
    replays: int,
    x,
    areas: dict[str, Any],
    replay_results: list[dict[str, Any]],
    replay_ratios: list[dict[str, Any]],
    average_replay_ratios: dict[str, Any],
    replay_totals: dict[str, Any],
    area_statistics: dict[str, Any],
    cdf_area_probabilities: dict[str, Any],
    equal_cdf_areas: dict[str, Any],
    method2_replay_totals: dict[str, Any],
) -> None:
    """Формирует подробный Markdown-отчёт для одного эксперимента."""
    config = normalize_distribution_config(distribution)
    x = list(x)
    orange_boundary = float(x[areas["orange"]["right"]])
    green_boundary = float(x[areas["green"]["left"]])
    p = cdf_area_probabilities
    p2 = equal_cdf_areas

    lines = [
        f"# Результаты эксперимента: {config['label']}",
        "",
        "## Параметры эксперимента",
        "",
        f"- Номер закона: `{config['id']}`",
        f"- Закон распределения: **{config['label']}** (`{config['key']}`)",
        f"- Параметры: `{config['params_text']}`",
        f"- Размер выборки: `N = {N}`",
        f"- Доля площади для метода 1: `m = {m}`",
        f"- Число повторов: `replays = {replays}`",
        f"- Всего сгенерировано элементов: `{N * replays}`",
        "",
        "## Границы областей метода 1",
        "",
        f"Оранжевая область определяется по плотности минимума `f_min(x)` и содержит приблизительно {m:.0%} её площади.",
        f"Зелёная область определяется по плотности максимума `f_max(x)` и содержит приблизительно {m:.0%} её площади.",
        "Синяя область — промежуток между правой границей оранжевой и левой границей зелёной областей; закраска выполняется по исходной плотности.",
        "",
        "| Область | Интервал |",
        "|---|---|",
        f"| Оранжевая | `x <= {orange_boundary:.12g}` |",
        f"| Синяя | `{orange_boundary:.12g} < x < {green_boundary:.12g}` |",
        f"| Зелёная | `x >= {green_boundary:.12g}` |",
        "",
        "## Теоретические вероятности метода 1",
        "",
        "Вероятности рассчитаны через функцию распределения выбранного закона `F(x)`: `P1 = F(x1)`, `P2 = F(x2) - F(x1)`, `P3 = 1 - F(x2)`.",
        "",
        "| Область | Граница | Значение CDF | Вероятность | Теоретическое число точек N·P |",
        "|---|---:|---:|---:|---:|",
        f"| Область 1 | x₁ = {p['orange_boundary']:.8g} | F(x₁) = {p['cdf_orange']:.6f} | {p['orange_probability']:.6f} | {N * p['orange_probability']:.4f} |",
        f"| Область 2 | x₁ < x < x₂ | между границами | {p['blue_probability']:.6f} | {N * p['blue_probability']:.4f} |",
        f"| Область 3 | x₂ = {p['green_boundary']:.8g} | F(x₂) = {p['cdf_green']:.6f} | {p['green_probability']:.6f} | {N * p['green_probability']:.4f} |",
        f"| **Сумма** | | | **{p['total']:.6f}** | **{N:.4f}** |",
        "",
        "## Метод 2: три равные области по CDF",
        "",
        "Границы выбраны по квантилям `F(x₁)=1/3` и `F(x₂)=2/3`, поэтому теоретическая вероятность каждой области равна 1/3.",
        "",
        "| Область | Граница / формула | Вероятность | Теоретическое число точек N·P |",
        "|---|---|---:|---:|",
        f"| Область 1 | `x <= {p2['orange_boundary']:.8g}` | {p2['orange_probability']:.6f} | {N * p2['orange_probability']:.4f} |",
        f"| Область 2 | `{p2['orange_boundary']:.8g} < x < {p2['green_boundary']:.8g}` | {p2['blue_probability']:.6f} | {N * p2['blue_probability']:.4f} |",
        f"| Область 3 | `x >= {p2['green_boundary']:.8g}` | {p2['green_probability']:.6f} | {N * p2['green_probability']:.4f} |",
        f"| **Сумма** | | **{p2['total']:.6f}** | **{N:.4f}** |",
        "",
        "## Сравнение фактических результатов методов",
        "",
        f"Фактические количества суммированы по {replays} replay (всего {N * replays} элементов).",
        "",
        "| Область | Метод 1: количество | Метод 1: доля | Метод 2: количество | Метод 2: доля |",
        "|---|---:|---:|---:|---:|",
        f"| Оранжевая | {replay_totals['orange_count']} | {replay_totals['orange_ratio']:.2%} | {method2_replay_totals['orange_count']} | {method2_replay_totals['orange_ratio']:.2%} |",
        f"| Синяя | {replay_totals['blue_count']} | {replay_totals['blue_ratio']:.2%} | {method2_replay_totals['blue_count']} | {method2_replay_totals['blue_ratio']:.2%} |",
        f"| Зелёная | {replay_totals['green_count']} | {replay_totals['green_ratio']:.2%} | {method2_replay_totals['green_count']} | {method2_replay_totals['green_ratio']:.2%} |",
        f"| **Всего** | **{replay_totals['total']}** | **100.00%** | **{method2_replay_totals['total']}** | **100.00%** |",
        "",
        "## Статистика областей метода 1",
        "",
        "Мат. ожидание — теоретическое ожидаемое число элементов в области для одной выборки; частота — общая фактическая доля по всем replay.",
        "",
        "| Область | Теоретическая вероятность | Мат. ожидание N·P | Фактическая частота | Среднее число точек | СКО | Дисперсия |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for area, label in (("orange", "Оранжевая"), ("blue", "Синяя"), ("green", "Зелёная")):
        s = area_statistics[area]
        lines.append(
            f"| {label} | {s['theoretical_probability']:.6f} | {s['mathematical_expectation']:.4f} | "
            f"{s['frequency']:.2%} | {s['mean']:.4f} | {s['standard_deviation']:.4f} | {s['variance']:.4f} |"
        )

    lines.extend([
        "",
        "## Средние доли по replay и сравнение с теорией",
        "",
        "| Область | Средняя доля n/N | Вероятность по CDF | Разница | Среднее число точек за replay |",
        "|---|---:|---:|---:|---:|",
    ])
    for area, label, ratio_key, prob_key in (
        ("orange", "Оранжевая", "orange_ratio", "orange_probability"),
        ("blue", "Синяя", "blue_ratio", "blue_probability"),
        ("green", "Зелёная", "green_ratio", "green_probability"),
    ):
        observed = average_replay_ratios[ratio_key]
        theoretical = p[prob_key]
        lines.append(
            f"| {label} | {observed:.6f} | {theoretical:.6f} | {observed - theoretical:+.6f} | {observed * N:.4f} |"
        )
    lines.append(f"| **Сумма** | **{average_replay_ratios['total']:.6f}** | **{p['total']:.6f}** | | **{N:.4f}** |")

    lines.extend([
        "",
        "## Доли точек по каждому replay",
        "",
        "| Replay | Область 1 n₁/N | Область 2 n₂/N | Область 3 n₃/N | Сумма |",
        "|---:|---:|---:|---:|---:|",
    ])
    for ratio in replay_ratios:
        ratio_sum = ratio["orange_ratio"] + ratio["blue_ratio"] + ratio["green_ratio"]
        lines.append(
            f"| {ratio['replay']} | {ratio['orange_ratio']:.6f} | {ratio['blue_ratio']:.6f} | "
            f"{ratio['green_ratio']:.6f} | {ratio_sum:.6f} |"
        )

    lines.extend([
        "",
        "## Количество элементов по каждому replay",
        "",
        "| Replay | Оранжевая | Синяя | Зелёная | Всего | Метод 2: область 1 | Метод 2: область 2 | Метод 2: область 3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in replay_results:
        lines.append(
            f"| {result['replay']} | {result['orange_count']} | {result['blue_count']} | {result['green_count']} | {result['total']} | "
            f"{result.get('method2_orange_count', '')} | {result.get('method2_blue_count', '')} | {result.get('method2_green_count', '')} |"
        )

    lines.extend([
        "",
        "## Состав сохранённых файлов",
        "",
        "- `density_min_max.png` — исходная плотность и плотности минимума/максимума.",
        "- `cdf_method_1.png` и `probabilities_method_1.png` — CDF и теоретические вероятности метода 1.",
        "- `cdf_method_2.png` и `probabilities_method_2.png` — CDF и теоретические вероятности метода 2.",
        "- `average_replay_shares.png` — средние доли по областям.",
        "- `all_replays_method_1.png` и `all_replays_method_2.png` — фактические доли по каждому replay.",
        "- `replays.csv` — фактические количества точек и метаданные закона для каждого replay.",
    ])

    report_file = output_dir / "report.md"
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
