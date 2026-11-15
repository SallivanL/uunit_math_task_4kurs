"""Гамма-модель, границы областей и повторения эксперимента."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc, logsumexp
from scipy.stats import gamma


@dataclass(frozen=True)
class Partition:
    boundaries: tuple[float, float]
    probabilities: np.ndarray
    cdf_levels: tuple[float, float]
    clean_expected_error: float
    x_max: float | None = None


@dataclass(frozen=True)
class AreaResult:
    p: float
    p_hat: float
    observed: int
    expected: float
    difference: float
    relative_deviation_percent: float
    pooled_contribution: float


@dataclass(frozen=True)
class MethodResult:
    counts: np.ndarray  # строки — повторения, столбцы — области слева направо
    fractions: np.ndarray
    areas: tuple[AreaResult, ...]
    errors: np.ndarray
    mean_error: float
    std_error: float
    pooled_error: float
    mean_standard_error: float | None
    expected_error: float
    bias_squared: float
    variance_coefficient: float


@dataclass(frozen=True)
class NoiseResult:
    alpha: float
    samples: np.ndarray
    methods: tuple[MethodResult, ...]


@dataclass(frozen=True)
class Curves:
    x: np.ndarray
    density: np.ndarray
    minimum_density: np.ndarray
    maximum_density: np.ndarray
    cdf: np.ndarray


@dataclass(frozen=True)
class Experiment:
    k_values: tuple[float, ...]
    theta: float
    N: int
    N_ref: int
    quotas: tuple[int, ...]
    koef: float
    seed: int
    partitions: tuple[Partition, ...]
    clean_samples: np.ndarray
    component_indices: np.ndarray
    noise_results: tuple[NoiseResult, ...]
    curves: Curves | None


@dataclass(frozen=True)
class SummaryRow:
    N: int
    alpha: float
    method: int
    mean_error: float
    std_error: float
    pooled_error: float
    clean_expected_error: float
    mean_standard_error: float | None
    expected_error: float
    bias_squared: float
    variance_coefficient: float
    seed: int
    replays: int
    N_ref: int


def _real(value, name, minimum, *, inclusive=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < minimum
        or (value == minimum and not inclusive)
    ):
        operator = '≥' if inclusive else '>'
        raise ValueError(f'{name} должно быть конечным числом {operator} {minimum}')
    return float(value)


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} должно быть целым числом ≥ {minimum}')
    return int(value)


def _unique(values, name):
    values = tuple(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f'{name} должен быть непустым списком без повторений')
    return values


def normalize_experiment_groups(massive_k, k_values):
    """Плоский massive_k задаёт одну смесь, вложенный — несколько групп."""
    if massive_k is None:
        groups = [(k,) for k in k_values]
    elif not isinstance(massive_k, (list, tuple)) or not massive_k:
        raise ValueError('massive_k должен быть непустым списком или None')
    elif any(isinstance(item, (list, tuple)) for item in massive_k):
        if not all(isinstance(group, (list, tuple)) for group in massive_k):
            raise ValueError('massive_k должен быть плоским списком либо списком групп')
        groups = massive_k
    else:
        groups = [massive_k]
    return _unique(
        (_unique((_real(k, 'k', 0) for k in group), 'Группа k') for group in groups),
        'Группы экспериментов',
    )


def calculate_mixed_sample_counts(k_values, N):
    """Распределяет N элементов поровну, остаток отдаёт первым компонентам."""
    if not k_values:
        raise ValueError('Нужна хотя бы одна компонента k')
    N = _integer(N, 'N', len(k_values))
    base, remainder = divmod(N, len(k_values))
    return [base + (index < remainder) for index in range(len(k_values))]


class GammaModel:
    """Независимые наблюдения с фиксированным числом элементов каждой компоненты."""

    def __init__(self, k_values, theta, N, *, quotas=None):
        self.k_values = _unique((_real(k, 'k', 0) for k in k_values), 'Группа k')
        self.theta = _real(theta, 'θ', 0)
        self.N = _integer(N, 'N', len(self.k_values))
        if quotas is None:
            quotas = calculate_mixed_sample_counts(self.k_values, self.N)
        self.quotas = tuple(_integer(count, 'Квота', 1) for count in quotas)
        if len(self.quotas) != len(self.k_values) or sum(self.quotas) != self.N:
            raise ValueError('Квоты должны соответствовать компонентам и в сумме давать N')
        self._counts = np.asarray(self.quotas)[:, None]
        self._weights = self._counts / self.N
        self._gamma = gamma(a=np.asarray(self.k_values)[:, None], scale=self.theta)

    def cdf(self, x):
        return np.sum(self._weights * self._gamma.cdf(x), axis=0)

    def _extreme_log_products(self, x):
        """Логарифмы F_max и S_min. В верхнем хвосте используем sf."""
        cdf = self._gamma.cdf(x)
        sf = self._gamma.sf(x)
        with np.errstate(divide='ignore'):
            log_cdf = np.where(cdf < 0.5, np.log(cdf), np.log1p(-np.minimum(sf, 0.5)))
            log_sf = np.where(cdf < 0.5, np.log1p(-np.minimum(cdf, 0.5)), np.log(sf))
        return np.stack((
            np.sum(self._counts * log_cdf, axis=0),
            np.sum(self._counts * log_sf, axis=0),
        ))

    def extreme_tail_masses(self, a, b):
        """P(min ≤ a) и P(max ≥ b) для чистого распределения."""
        log_max, log_min_survival = self._extreme_log_products([a, b])
        return float(-np.expm1(log_min_survival[0])), float(-np.expm1(log_max[1]))

    def _quantile_root(self, function, upper):
        # Решаем в единицах масштаба θ, поэтому точность не зависит от θ.
        return self.theta * brentq(
            lambda value: function(value * self.theta), 0.0, upper / self.theta,
            xtol=np.finfo(float).tiny, rtol=1e-14, maxiter=1200,
        )

    def quantile(self, probability):
        component_quantiles = self._gamma.ppf(probability)
        if len(self.k_values) == 1:
            return float(component_quantiles[0, 0])
        upper = float(np.max(component_quantiles))
        # При практически совпадающих CDF всех компонентов возможна ошибка округления.
        if self.cdf(upper)[0] <= probability:
            return upper
        return self._quantile_root(lambda x: self.cdf(x)[0] - probability, upper)

    def _max_plot_upper(self, max_tail):
        component_tail = -np.expm1(np.log1p(-max_tail) / self.N)
        return float(np.max(self._gamma.isf(component_tail)))

    def _minimum_quantile(self, probability):
        component_tail = -np.expm1(np.log1p(-probability) / self.N)
        if len(self.k_values) == 1:
            return float(self._gamma.ppf(component_tail)[0, 0])
        upper = float(np.max(self._gamma.ppf(component_tail)))
        target = np.log1p(-probability)
        objective = lambda x: target - self._extreme_log_products(x)[1, 0]
        if objective(upper) <= 0:
            return upper
        return self._quantile_root(objective, upper)

    def extreme_boundaries(self, koef):
        koef = _real(koef, 'koef', 0)
        if koef >= 1:
            raise ValueError('koef должно быть меньше 1')
        a = self._minimum_quantile(koef)
        if len(self.k_values) == 1:
            p = -np.expm1(np.log1p(-koef) / self.N)
            b = float(self._gamma.isf(p)[0, 0])
        else:
            upper = self._max_plot_upper(koef)
            target = np.log1p(-koef)
            objective = lambda x: self._extreme_log_products(x)[0, 0] - target
            b = upper if objective(upper) <= 0 else self._quantile_root(objective, upper)
        if not 0 < a < b < np.inf:
            raise ValueError(
                f'Метод 1: границы не образуют три области при N={self.N}, koef={koef}. '
                'Увеличьте N или уменьшите koef.'
            )
        return a, b

    def component_probabilities(self, boundaries):
        """Вероятности областей чистого распределения по компонентам."""
        a, b = boundaries
        if not 0 < a < b < np.inf:
            raise ValueError('Границы должны быть конечными и удовлетворять 0 < a < b')
        cdf = self._gamma.cdf([a, b])
        sf = self._gamma.sf([a, b])
        # Разность функций выживания точнее, когда обе CDF близки к единице.
        middle = np.where(cdf[:, 0] < 0.5, cdf[:, 1] - cdf[:, 0], sf[:, 0] - sf[:, 1])
        return np.column_stack((cdf[:, 0], middle, sf[:, 1]))

    def partition(self, boundaries, *, x_max=None):
        """Вероятности и E[D] при α=0 с учётом фиксированных квот."""
        by_component = self.component_probabilities(boundaries)
        probabilities = np.sum(self._weights * by_component, axis=0)
        if np.any(probabilities <= 0) or not np.isclose(
            probabilities.sum(), 1.0, rtol=0, atol=1e-12
        ):
            raise ValueError('Вероятности областей должны быть положительными и давать сумму 1')
        _, variance = self.error_terms(probabilities, by_component)
        expected_error = variance / self.N
        levels = tuple(float(value) for value in self.cdf(boundaries))
        return Partition(boundaries, probabilities, levels, expected_error, x_max)

    def noisy_component_probabilities(self, boundaries, alpha):
        """Интегрирует Gamma CDF по стандартному нормальному шуму.

        При положительных a и b обрезка в нуле оставляет все значения ≤ 0
        в левой области, поэтому вероятности областей можно считать по X + αZ.
        """
        if alpha == 0:
            return self.component_probabilities(boundaries)
        return np.array([
            _integrate_noisy_areas(k, self.theta, boundaries, alpha) for k in self.k_values
        ])

    def error_terms(self, p, by_component):
        """Смещение A и коэффициент дисперсии B для E[D] = A + B/N."""
        q = np.sum(self._weights * by_component, axis=0)
        bias_squared = float(relative_error(q, p))
        # Складываем две другие области, чтобы сохранить малый хвост при q_ij ≈ 1.
        complement = by_component[:, [1, 0, 0]] + by_component[:, [2, 2, 1]]
        scaled_variance = (by_component / p) * (complement / p)
        return bias_squared, float(np.sum(self._weights * scaled_variance))

    def partitions(self, koef):
        # Сначала вычисляем интервалы. Сетка для рисунков добавляется позже.
        x_max = self.quantile(0.9999)
        return (
            self.partition(self.extreme_boundaries(koef)),
            self.partition((self.quantile(1 / 3), self.quantile(2 / 3))),
            self.partition((x_max / 3, 2 * x_max / 3), x_max=x_max),
        )

    def curves(self, partitions, points=5000):
        points = _integer(points, 'Число точек графика', 30)
        method3_max = partitions[2].x_max
        upper = max(self._max_plot_upper(0.0001), method3_max, partitions[0].boundaries[1])
        minimum_focus = self._minimum_quantile(0.995)
        # Не включаем x=0 в сетку плотности: при k < 1 там возможна особенность.
        low = np.geomspace(minimum_focus * 1e-6, minimum_focus, points // 3)
        rest = np.linspace(minimum_focus, upper, points - len(low) + 1)[1:]
        landmarks = [boundary for part in partitions for boundary in part.boundaries]
        x = np.unique(np.r_[low, rest, landmarks, method3_max])
        log_pdf = self._gamma.logpdf(x)
        cdf = self._gamma.cdf(x)
        sf = self._gamma.sf(x)
        density = np.exp(logsumexp(log_pdf + np.log(self._weights), axis=0))
        log_cdf = np.where(cdf < 0.5, self._gamma.logcdf(x), np.log1p(-np.minimum(sf, 0.5)))
        log_sf = self._gamma.logsf(x)

        def extreme_density(log_factors):
            # Нулевая степень даёт множитель 1 даже при log(F_i) = −∞.
            contributions = []
            for index, count in enumerate(self.quotas):
                other = np.arange(len(self.quotas)) != index
                combined = np.sum(self._counts[other] * log_factors[other], axis=0)
                if count > 1:
                    combined += (count - 1) * log_factors[index]
                contributions.append(np.log(count) + log_pdf[index] + combined)
            return np.exp(logsumexp(contributions, axis=0))

        return Curves(x, density, extreme_density(log_sf), extreme_density(log_cdf), self.cdf(x))

    def draw(self, replays, rng):
        # Прямой набор n_i независимых Gamma-точек эквивалентен выбору n_i
        # случайных индексов без возвращения из N независимых точек компоненты.
        samples = np.concatenate([
            rng.gamma(k, self.theta, size=(replays, count))
            for k, count in zip(self.k_values, self.quotas)
        ], axis=1)
        labels = np.broadcast_to(np.repeat(np.arange(len(self.quotas)), self.quotas), samples.shape)
        order = rng.permuted(np.broadcast_to(np.arange(self.N), samples.shape), axis=1)
        return np.take_along_axis(samples, order, axis=1), np.take_along_axis(labels, order, axis=1)


def _integrate_noisy_areas(k, theta, boundaries, alpha):
    a, b = boundaries
    normalizer = 1 / np.sqrt(2 * np.pi)

    def integrand(z, area):
        left = max(0.0, (a - alpha * z) / theta)
        right = max(0.0, (b - alpha * z) / theta)
        if area == 0:
            probability = gammainc(k, left)
        elif area == 2:
            probability = gammaincc(k, right)
        elif gammainc(k, left) < 0.5:
            probability = gammainc(k, right) - gammainc(k, left)
        else:
            probability = gammaincc(k, left) - gammaincc(k, right)
        return probability * normalizer * np.exp(-0.5 * z * z)

    # Разбиение в точках излома CDF и в центральной части нормальной плотности.
    breakpoints = sorted({-8.0, 0.0, 8.0} | {x / alpha for x in boundaries if x < 12 * alpha})
    intervals = [-np.inf, *breakpoints, np.inf]
    probabilities = [
        sum(quad(integrand, lo, hi, args=(area,), epsabs=1e-13, epsrel=1e-10)[0]
            for lo, hi in zip(intervals, intervals[1:]))
        for area in range(3)
    ]
    if not np.isclose(sum(probabilities), 1.0, rtol=0, atol=1e-10):
        raise ValueError('Вероятности областей с шумом должны давать сумму 1')
    return probabilities


def apply_noise(samples, standard_noise, alpha):
    """Y = max(0, X + αZ). Для α=0 сохраняет чистые наблюдения."""
    alpha = _real(alpha, 'α', 0, inclusive=True)
    if alpha == 0:
        return samples
    with np.errstate(over='ignore', invalid='ignore'):
        noisy = np.maximum(0.0, samples + alpha * standard_noise)
    if not np.all(np.isfinite(noisy)):
        raise ValueError('Зашумлённая выборка содержит неконечные значения')
    return noisy


def classify_samples(samples, boundaries):
    """Для каждого повторения возвращает счётчики [x≤a, a<x<b, x≥b]."""
    a, b = boundaries
    if not a < b or not np.all(np.isfinite(samples)):
        raise ValueError('Для классификации нужны упорядоченные границы и конечные наблюдения')
    left = np.count_nonzero(samples <= a, axis=1)
    right = np.count_nonzero(samples >= b, axis=1)
    return np.column_stack((left, samples.shape[1] - left - right, right))


def relative_error(observed_fractions, probabilities):
    """D = Σ ((p − p̂) / p)²."""
    return np.sum(((probabilities - observed_fractions) / probabilities) ** 2, axis=-1)


def evaluate_method(samples, partition, error_terms):
    counts = classify_samples(samples, partition.boundaries)
    fractions = counts / samples.shape[1]
    errors = relative_error(fractions, partition.probabilities)
    areas = []
    for p, observed in zip(partition.probabilities, counts.sum(axis=0)):
        p_hat = observed / samples.size
        expected = p * samples.size
        relative = (p_hat - p) / p
        areas.append(AreaResult(float(p), float(p_hat), int(observed), float(expected),
                                float(observed - expected), float(100 * relative),
                                float(relative**2)))
    pooled = sum(area.pooled_contribution for area in areas)
    std = float(errors.std())
    standard_error = std / np.sqrt(len(errors) - 1) if len(errors) > 1 else None
    bias_squared, variance_coefficient = error_terms
    return MethodResult(
        counts, fractions, tuple(areas), errors, float(errors.mean()), std, pooled,
        standard_error, bias_squared + variance_coefficient / samples.shape[1],
        bias_squared, variance_coefficient,
    )


def run_experiments(
    *, k_values, massive_k, theta_values, N_values, alpha_values, koef, replays, seed,
    N_ref=20, with_curves=True,
):
    """Генерирует по одному результату для каждой комбинации группы k, θ и N."""
    groups = normalize_experiment_groups(massive_k, k_values)
    thetas = _unique((_real(value, 'θ', 0) for value in theta_values), 'Значения θ')
    sizes = _unique((_integer(value, 'N', 1) for value in N_values), 'Значения N')
    N_ref = _integer(N_ref, 'N_ref', 1)
    alphas = _unique((_real(alpha, 'α', 0, inclusive=True) for alpha in alpha_values), 'Значения α')
    replays = _integer(replays, 'Число повторений', 1)
    seed = _integer(seed, 'Seed', 0) if seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)

    quotas_by_group = {}
    for group in groups:
        reference_quotas = calculate_mixed_sample_counts(group, N_ref)
        quotas_by_group[group] = {}
        for N in sizes:
            counts = [count * N for count in reference_quotas]
            if any(count % N_ref for count in counts):
                raise ValueError(f'N={N} не сохраняет точные пропорции квот {reference_quotas} '
                                 f'группы k={group} при N_ref={N_ref}.')
            quotas_by_group[group][N] = [count // N_ref for count in counts]

    for group in groups:
        for theta in thetas:
            reference = GammaModel(group, theta, N_ref)
            reference_parts = reference.partitions(koef)
            curves = reference.curves(reference_parts) if with_curves else None
            terms = {
                alpha: tuple(reference.error_terms(
                    part.probabilities, reference.noisy_component_probabilities(part.boundaries, alpha),
                ) for part in reference_parts)
                for alpha in alphas
            }
            for N in sizes:
                model = GammaModel(group, theta, N, quotas=quotas_by_group[group][N])
                partitions = tuple(
                    Partition(part.boundaries, part.probabilities, part.cdf_levels,
                              part.clean_expected_error * N_ref / N, part.x_max)
                    for part in reference_parts
                )
                clean, components = model.draw(replays, rng)
                standard_noise = rng.standard_normal(clean.shape)
                results = []
                for alpha in alphas:
                    samples = apply_noise(clean, standard_noise, alpha)
                    methods = tuple(evaluate_method(samples, part, method_terms)
                                    for part, method_terms in zip(partitions, terms[alpha]))
                    results.append(NoiseResult(alpha, samples, methods))
                yield Experiment(group, theta, N, N_ref, model.quotas, float(koef), seed, partitions,
                                 clean, components, tuple(results), curves)


def summarize(experiment):
    return [
        SummaryRow(experiment.N, scenario.alpha, index, method.mean_error, method.std_error,
                   method.pooled_error, part.clean_expected_error, method.mean_standard_error,
                   method.expected_error, method.bias_squared, method.variance_coefficient,
                   experiment.seed, len(experiment.clean_samples), experiment.N_ref)
        for scenario in experiment.noise_results
        for index, (method, part) in enumerate(zip(scenario.methods, experiment.partitions), 1)
    ]
