import unittest

from steering.hyperparameters import (
    AVERAGING_ALPHA_BY_VALUE,
    AVERAGING_DEFAULT_ALPHA,
    DEFAULT_LAYER_RANGE,
    PROBE_P0,
    PROJECTION_COEFF,
    get_averaging_alpha,
    get_backend_kwargs,
    to_repe_hidden_layers,
)
from value_selection.values import ALL_VALUES


class HyperparameterAlignmentTests(unittest.TestCase):
    def test_scalar_defaults_match_paper(self):
        self.assertEqual(DEFAULT_LAYER_RANGE, "10-25")
        self.assertEqual(PROJECTION_COEFF, 1.0)
        self.assertEqual(AVERAGING_DEFAULT_ALPHA, 0.5)
        self.assertEqual(PROBE_P0, 0.9)

    def test_prior_work_averaging_coefficients(self):
        self.assertEqual(len(AVERAGING_ALPHA_BY_VALUE), 10)
        self.assertEqual(set(AVERAGING_ALPHA_BY_VALUE), {slug for slug, _, _ in ALL_VALUES[:10]})
        self.assertEqual(get_averaging_alpha("achievement"), 0.3)
        self.assertEqual(get_averaging_alpha("self-direction"), 0.6)
        self.assertEqual(get_averaging_alpha("benevolence"), 0.08)
        self.assertEqual(get_averaging_alpha("universalism"), 0.215)

    def test_extended_value_uses_shared_averaging_fallback(self):
        self.assertEqual(get_averaging_alpha("justice"), 0.5)
        self.assertEqual(get_averaging_alpha("justice", default_alpha=0.75), 0.75)

    def test_backend_kwargs_propagate_paper_aligned_configuration(self):
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.assertEqual(
            get_backend_kwargs("probe_calibrated", model),
            {"p0": 0.9, "model_name": "Meta-Llama-3-8B-Instruct", "use_gate": False},
        )
        self.assertEqual(
            get_backend_kwargs("averaging_caa", model),
            {"default_alpha": 0.5, "model_name": "Meta-Llama-3-8B-Instruct"},
        )
        self.assertEqual(get_backend_kwargs("projection_pca", model), {"coeff": 1.0})

    def test_repe_mapping_preserves_physical_blocks(self):
        self.assertEqual(to_repe_hidden_layers(range(10, 26), 32), list(range(-22, -6)))
        self.assertEqual(to_repe_hidden_layers([0, 31], 32), [-32, -1])

    def test_repe_mapping_rejects_invalid_physical_block(self):
        with self.assertRaises(ValueError):
            to_repe_hidden_layers([32], 32)


if __name__ == "__main__":
    unittest.main()
