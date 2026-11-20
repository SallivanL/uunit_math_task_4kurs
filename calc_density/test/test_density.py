import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import exponnorm, gamma

from calc import (
    GammaModel, apply_noise, classify_samples, normalize_experiment_groups, run_experiments,
)
from main import main
from plotting import plot_comparison, plot_comparison_three_methods, plot_replay_histogram


def make_case(**options):
    config = dict(k_values=[1], massive_k=None, theta_values=[1], N_values=[20],
                  alpha_values=[0, 0.5, 2], koef=0.7, replays=6, seed=20261115)
    case, = run_experiments(**(config | options))
    return case


class CalculationTests(unittest.TestCase):
    def test_single_extreme_quantiles_match_full_probability_mass(self):
        model = GammaModel((8,), theta=1, N=20)
        left, right = model.extreme_boundaries(0.7)
        self.assertAlmostEqual(left, 4.117234711108874, places=10)
        self.assertAlmostEqual(right, 12.849196788193877, places=10)
        for n in (20, 1000):
            model = GammaModel((8,), 1, n)
            a, b = model.extreme_boundaries(0.7)
            self.assertAlmostEqual(1 - gamma.sf(a, 8)**n, 0.7, places=12)
            self.assertAlmostEqual(1 - gamma.cdf(b, 8)**n, 0.7, places=12)
            self.assertGreaterEqual(gamma.cdf(model.curves(model.partitions(.7)).x[-1], 8)**n,
                                    0.9999 - 1e-12)

    def test_mixed_extremes_and_base_cdf_use_the_actual_quotas(self):
        model = GammaModel((1, 8, 15), theta=2, N=20)
        self.assertEqual(model.quotas, (7, 7, 6))
        shapes = np.array(model.k_values)
        counts = np.array(model.quotas)
        first, second, third = model.partitions(0.7)
        a, b = first.boundaries
        self.assertAlmostEqual(1 - np.prod(gamma.sf(a, shapes, scale=2)**counts), 0.7)
        self.assertAlmostEqual(1 - np.prod(gamma.cdf(b, shapes, scale=2)**counts), 0.7)
        for part in (first, second, third):
            self.assertTrue(np.all(part.probabilities > 0))
            self.assertAlmostEqual(part.probabilities.sum(), 1)
        np.testing.assert_allclose(second.probabilities, [1/3] * 3, atol=1e-12)
        self.assertAlmostEqual(
            np.dot(counts / 20, gamma.cdf(third.x_max, shapes, scale=2)), 0.9999,
        )
        np.testing.assert_allclose(np.diff([0, *third.boundaries, third.x_max]), third.x_max / 3)

    def test_curve_grid_includes_exact_cutoffs_with_fractional_shape(self):
        for group in [(0.4,), (0.4, 8)]:
            model = GammaModel(group, 1, 1000)
            parts = model.partitions(0.7)
            for points in (30, 5000):
                curves = model.curves(parts, points)
                for part in parts:
                    self.assertTrue(all(boundary in curves.x for boundary in part.boundaries))
                for density in (curves.density, curves.minimum_density, curves.maximum_density):
                    self.assertTrue(np.isfinite(density).all())
            np.testing.assert_allclose(
                model.extreme_tail_masses(*parts[0].boundaries), [0.7, 0.7], atol=1e-12,
            )
        case = make_case(k_values=[0.4])
        self.assertEqual(case.k_values, (0.4,))

    def test_replays_share_noisy_samples_and_keep_quotas(self):
        case = make_case(massive_k=[1, 8, 15])
        reordered = make_case(massive_k=[[1, 8, 15]], alpha_values=[2, 0.5, 0])
        np.testing.assert_array_equal(case.clean_samples, reordered.clean_samples)
        for labels in case.component_indices:
            np.testing.assert_array_equal(np.bincount(labels), case.quotas)
        clean, moderate, strong = case.noise_results
        np.testing.assert_array_equal(clean.samples, case.clean_samples)
        np.testing.assert_array_equal(strong.samples, reordered.noise_results[0].samples)
        inferred_noise = (moderate.samples - clean.samples) / moderate.alpha
        mask = moderate.samples > 0
        np.testing.assert_allclose(strong.samples[mask],
                                  np.maximum(0, clean.samples + 2 * inferred_noise)[mask])
        self.assertGreater(np.count_nonzero(strong.samples == 0), 0)
        for scenario in case.noise_results:
            for part, method in zip(case.partitions, scenario.methods):
                for sample, counts in zip(scenario.samples, method.counts):
                    a, b = part.boundaries
                    buckets = [0 if value <= a else 2 if value >= b else 1 for value in sample]
                    np.testing.assert_array_equal(counts, np.bincount(buckets, minlength=3))
                np.testing.assert_allclose(
                    method.errors, (((part.probabilities - method.counts / case.N)
                                     / part.probabilities)**2).sum(axis=1),
                )

    def test_boundaries_zero_and_outside_the_plot(self):
        np.testing.assert_array_equal(
            classify_samples(np.array([[0, 1, 2, 3, 100]]), (1, 3)), [[2, 1, 2]],
        )
        np.testing.assert_array_equal(
            apply_noise(np.array([[1., 2.]]), np.array([[-2., 1.]]), 1), [[0, 3]],
        )

    def test_noisy_exponential_probabilities_match_exgaussian_cdf(self):
        model = GammaModel((1,), theta=2, N=20)
        a, b = model.partitions(0.7)[0].boundaries
        for alpha in (0.2, 2.0):
            distribution = exponnorm(2 / alpha, scale=alpha)
            left, through_middle = distribution.cdf([a, b])
            expected = [left, through_middle - left, distribution.sf(b)]
            observed, = model.noisy_component_probabilities((a, b), alpha)
            np.testing.assert_allclose(observed, expected, rtol=1e-9)

    def test_fixed_mixed_boundaries_and_expected_error_at_different_sizes(self):
        small, large = run_experiments(
            k_values=(), massive_k=[1, 8, 15], theta_values=[1], N_values=[20, 100],
            N_ref=20, alpha_values=[0, 0.5, 2], koef=0.7, replays=2, seed=24, with_curves=False,
        )
        self.assertEqual(small.quotas, (7, 7, 6))
        self.assertEqual(large.quotas, (35, 35, 30))
        for small_part, large_part in zip(small.partitions, large.partitions):
            self.assertEqual(small_part.boundaries, large_part.boundaries)
            np.testing.assert_array_equal(small_part.probabilities, large_part.probabilities)
        for first, second in zip(small.noise_results, large.noise_results):
            for lower, higher in zip(first.methods, second.methods):
                self.assertGreater(lower.expected_error, higher.expected_error)
                self.assertAlmostEqual(lower.bias_squared, higher.bias_squared)
                self.assertAlmostEqual(lower.variance_coefficient, higher.variance_coefficient)
                if first.alpha == 0:
                    self.assertEqual(lower.bias_squared, 0)
                    self.assertAlmostEqual(lower.expected_error / higher.expected_error, 5)

    def test_invalid_parameters_and_overlapping_extreme_areas(self):
        for options in ({'k_values': [0]}, {'theta_values': [float('nan')]}, {'N_values': [3.5]},
                        {'alpha_values': [-1]}, {'replays': 0}, {'koef': 1}, {'seed': -1},
                        {'massive_k': [1, 2, 3], 'N_values': [2]},
                        {'massive_k': [1, 8], 'N_values': [21]}, {'N_ref': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_case(**options)
        with self.assertRaisesRegex(ValueError, 'границы'):
            GammaModel((1,), 1, 1).partitions(0.7)
        with self.assertRaises(ValueError):
            normalize_experiment_groups([1, [8, 15]], (1,))
        with self.assertRaises(ValueError):
            normalize_experiment_groups([[1, 8], [1.0, 8.0]], (1,))


class ReportTests(unittest.TestCase):
    def test_main_exports_the_same_data_to_csv_markdown_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch('main.plot_comparison_three_methods', wraps=plot_comparison_three_methods)
                as comparison_plot,
                patch('main.plot_replay_histogram', wraps=plot_replay_histogram) as initial_histogram,
                patch('main.plot_comparison', wraps=plot_comparison) as expected_plot,
            ):
                root = main(massive_k=[[1, 8, 15], [2, 4]], N_values=(20,), theta_values=(1, 2),
                            alpha_values=(1, 0), replays=2, seed=24, results_root=tmp)
            self.assertEqual(root, Path(tmp))
            group = root / 'k_1_8_15'
            path = group / 'alpha_0' / 'N_20'
            noisy_path = group / 'alpha_1' / 'N_20'
            suffix = '_k_1_8_15_theta_1_N_20'
            base_graphs = {'initial_graphs', 'initial_histogram', 'method_3_distribution'}
            base_graphs.update(
                f'method_{method}_{kind}' for method in (1, 2, 3)
                for kind in ('cdf', 'theoretical_probabilities', 'replay_histograms')
            )
            for graph in base_graphs:
                self.assertTrue((path / f'{graph}{suffix}.png').is_file(), graph)
                self.assertTrue((noisy_path / f'{graph}{suffix}.png').is_file(), graph)
            self.assertTrue((path / 'report_k_1_8_15_theta_2_N_20.md').is_file())
            self.assertTrue((group / 'alpha_0' / 'comparison_three_methods_theta_1.png').is_file())
            self.assertTrue((group / 'noise_comparison_theta_1.png').is_file())
            self.assertEqual({item.name for item in group.iterdir() if item.is_dir()},
                             {'alpha_0', 'alpha_1'})

            replays = self.read_csv(path / f'replays{suffix}.csv')
            areas = self.read_csv(path / f'chi_squared_comparison{suffix}.csv')
            clean = self.read_csv(path / f'samples{suffix}.csv')
            samples = self.read_csv(noisy_path / f'samples{suffix}.csv')
            notes = (path / f'report{suffix}.md').read_text()
            summary = self.read_csv(group / 'alpha_0' / 'comparison_three_methods_theta_1.csv')
            diagnostics = self.read_csv(group / 'alpha_1' / 'comparison_by_N_alpha_1_theta_1.csv')
            all_groups = self.read_csv(root / 'comparison_all_groups_theta_1.csv')
            self.assertEqual({row['k_group'] for row in all_groups}, {'1_8_15', '2_4'})
            group_coefficients = self.read_csv(group / 'comparison_theta_1.csv')
            self.assertEqual(list(group_coefficients[0]),
                             ['alpha', 'N', 'method_1_D', 'method_2_D', 'method_3_D'])
            self.assertEqual([float(row['alpha']) for row in group_coefficients], [0, 1])
            for row in group_coefficients:
                matching = [item for item in all_groups
                            if item['k_group'] == '1_8_15' and item['N'] == row['N']
                            and item['alpha'] == row['alpha']]
                self.assertEqual(len(matching), 3)
                for item in matching:
                    self.assertEqual(float(row[f"method_{item['method']}_D"]),
                                     float(item['pooled_error']))
            self.assertTrue((root / 'comparison_all_groups_theta_1_alpha_1.png').is_file())
            self.assertTrue((root / 'comparison_all_groups_theta_1_N_20.png').is_file())
            coefficients = self.read_csv(root / 'comparison_all_groups_theta_1_N_20.csv')
            self.assertEqual(len(coefficients), 4)
            for row in all_groups:
                matching = next(item for item in coefficients
                                if item['k_group'] == row['k_group'] and item['alpha'] == row['alpha'])
                self.assertEqual(float(matching[f"method_{row['method']}_D"]),
                                 float(row['pooled_error']))
            self.assertEqual({key: summary[0][key] for key in ('alpha', 'seed', 'replays', 'N_ref')},
                             {'alpha': '0.0', 'seed': '24', 'replays': '2', 'N_ref': '20'})
            expected_call = next(
                call for call in expected_plot.call_args_list
                if call.kwargs['x_axis'] == 'N' and call.kwargs['fixed_value'] == 1
                and call.kwargs['theta'] == 1 and call.kwargs['k_values'] == (1, 8, 15)
            )
            expected_chart = plot_comparison(*expected_call.args, **expected_call.kwargs)
            noisy_notes = (noisy_path / f'report{suffix}.md').read_text()
            for ax, row in zip(expected_chart.axes, diagnostics):
                self.assertAlmostEqual(ax.lines[0].get_ydata()[0], float(row['expected_error']))
                self.assertAlmostEqual(ax.lines[1].get_ydata()[0], float(row['mean_error']))
                self.assertIn(f"| {float(row['expected_error']):.12g} |", noisy_notes)
            self.assertEqual(len(replays), 2)
            self.assertEqual(len(samples), 40)
            self.assertEqual(len(areas), 9)
            self.assertEqual(list(replays[0]), [
                'replay', 'orange_count', 'blue_count', 'green_count',
                'method2_orange_count', 'method2_blue_count', 'method2_green_count',
                'method3_orange_count', 'method3_blue_count', 'method3_green_count', 'total',
                'selected_from_k_1', 'selected_from_k_15', 'selected_from_k_8',
            ])
            self.assertEqual([s['clean'] for s in samples], [s['noisy'] for s in clean])
            self.assertIn('## 2. Сравнение теоретического и фактического количества точек', notes)
            self.assertIn('Квоты смешанной выборки', notes)
            comparison_call = next(
                call for call in comparison_plot.call_args_list
                if call.kwargs['alpha'] == 0 and call.kwargs['theta'] == 1
                and call.kwargs['k_values'] == (1, 8, 15)
            )
            compare_figure = plot_comparison_three_methods(*comparison_call.args,
                                                           **comparison_call.kwargs)
            for number in (1, 2, 3):
                method_areas = [area for area in areas if area['method'] == f'Метод {number}']
                pooled = sum(float(area['chi_squared_contribution']) for area in method_areas)
                self.assertAlmostEqual(float(summary[0][f'method_{number}_chi_squared']), pooled)
                self.assertIn(f'**Метод {number}: итог χ²**', notes)
                self.assertAlmostEqual(compare_figure.axes[0].lines[number - 1].get_ydata()[0],
                                       pooled)
            histogram_call = next(
                call for call in initial_histogram.call_args_list
                if call.args[0].theta == 1 and call.args[1].alpha == 0
                and call.args[0].k_values == (1, 8, 15)
            )
            histogram = plot_replay_histogram(*histogram_call.args)
            np.testing.assert_allclose(
                [bar.get_height() for bar in histogram.axes[0].patches],
                [float(area['empirical_probability_p_hat']) for area in areas[:3]],
            )

    @staticmethod
    def read_csv(path: Path):
        with path.open(newline='', encoding='utf-8') as handle:
            return list(csv.DictReader(handle))


if __name__ == '__main__':
    unittest.main()
