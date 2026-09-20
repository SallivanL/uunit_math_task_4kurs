import numpy as np
from scipy.stats import gamma


# ===== МИНИМАЛЬНЫЙ КВАНТИЛЬ (ДЛЯ ω=0) =====
OMEGA_MIN = 0.005
EPS = 1e-8


def effective_omega(omega):
    if omega == 0:
        return OMEGA_MIN
    return omega


# ===== ГЕНЕРАЦИЯ ВЫБОРКИ (БЕЗ ШУМА) =====
def generate_sample(k, theta, N):
    return np.random.gamma(shape=k, scale=theta, size=N)


def empirical_cdf(sample):
    x = np.sort(sample)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x, y


def gamma_cdf(x, k, theta):
    return gamma.cdf(x, a=k, scale=theta)


# ===== ИНТЕГРАЛЬНАЯ МЕТРИКА =====
def max_diff_trapezoid(x, F, F_hat):
    diff = np.abs(F - F_hat)

    integral = 0.0
    max_integral = 0.0

    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        area = (diff[i] + diff[i - 1]) / 2 * dx
        integral += area
        max_integral = max(max_integral, integral)

    return max_integral


# ===== ИНТЕРПОЛЯЦИЯ =====
def find_x_interp(x, F, target):
    for i in range(1, len(x)):
        if F[i] >= target:
            x1, x2 = x[i - 1], x[i]
            y1, y2 = F[i - 1], F[i]

            if y2 == y1:
                return x2

            return x1 + (target - y1) * (x2 - x1) / (y2 - y1)

    return x[-1]


# ===== ТЕОРИЯ =====
def theoretical_interval(k, theta, N, omega):
    omega_eff = effective_omega(omega)

    x = np.linspace(0, 10 * k * theta, 10000)
    F = gamma_cdf(x, k, theta)

    F_max = F ** N
    F_min = 1 - (1 - F) ** N

    x_left = find_x_interp(x, F_min, omega_eff)
    x_right = find_x_interp(x, F_max, 1 - omega_eff)

    return x_right - x_left


# ===== ЭКСТРЕМУМЫ (С ДВОЙНЫМ ALPHA) =====
def generate_extreme_samples(k, theta, N, M_inner, alpha_pair):
    alpha1, alpha2 = alpha_pair

    max_samples = np.empty(M_inner)
    min_samples = np.empty(M_inner)

    for i in range(M_inner):
        subsample = np.random.gamma(shape=k, scale=theta, size=N)

        if alpha1 == alpha2:
            if alpha1 > 0:
                noise = np.random.normal(0, alpha1, size=N)
                subsample = np.clip(subsample + noise, 0, None)

        else:
            # половина с alpha1, половина с alpha2
            half = N // 2

            noise1 = np.random.normal(0, alpha1, size=half)
            noise2 = np.random.normal(0, alpha2, size=N - half)

            noise = np.concatenate([noise1, noise2])

            # перемешиваем, чтобы убрать порядок
            np.random.shuffle(noise)

            subsample = np.clip(subsample + noise, 0, None)

        max_samples[i] = np.max(subsample)
        min_samples[i] = np.min(subsample)

    return max_samples, min_samples


# ===== ОСНОВНОЙ ЭКСПЕРИМЕНТ =====
def run_experiment(k, theta, N, alpha_pair, omega):
    omega_eff = effective_omega(omega)

    # ===== ОСНОВНАЯ ВЫБОРКА =====
    sample = generate_sample(k, theta, N)

    x, F_hat = empirical_cdf(sample)
    F = gamma_cdf(x, k, theta)

    D_F = max_diff_trapezoid(x, F, F_hat)

    # ===== ЭКСТРЕМУМЫ =====
    M_inner = N * 1
    max_samples, min_samples = generate_extreme_samples(
        k, theta, N, M_inner, alpha_pair
    )

    x_max, F_hat_max = empirical_cdf(max_samples)
    x_min, F_hat_min = empirical_cdf(min_samples)

    F_max = gamma_cdf(x_max, k, theta) ** N
    F_min = 1 - (1 - gamma_cdf(x_min, k, theta)) ** N

    D_Fmax = max_diff_trapezoid(x_max, F_max, F_hat_max)
    D_Fmin = max_diff_trapezoid(x_min, F_min, F_hat_min)

    # ===== ЭМПИРИЧЕСКИЙ ИНТЕРВАЛ =====
    x_left_emp = find_x_interp(x_min, F_hat_min, omega_eff)
    x_right_emp = find_x_interp(x_max, F_hat_max, 1 - omega_eff)
    d_emp = x_right_emp - x_left_emp

    # ===== ТЕОРИЯ =====
    d_true = theoretical_interval(k, theta, N, omega)

    # ===== МЕТРИКА (КОНУС) =====
    R = d_emp / max(d_true, EPS)

    return D_F, D_Fmax, D_Fmin, R