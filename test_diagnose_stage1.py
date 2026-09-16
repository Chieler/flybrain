"""Checks for the reproducible Stage 1 upstream diagnostic."""

import importlib
import importlib.util
import math
import unittest

import numpy as np
import scipy.sparse as sp

from brain import Brain, RateParams


class TestDriveComponents(unittest.TestCase):
    def test_partition_sums_to_full_recurrent_drive(self):
        self.assertIsNotNone(
            importlib.util.find_spec("diagnose_stage1"),
            "diagnose_stage1.py is missing",
        )
        diagnostic = importlib.import_module("diagnose_stage1")
        weights = sp.csr_array(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [2, -1, 3, 0],
                ],
                dtype=np.float32,
            )
        )
        brain = Brain(weights, {}, {}, RateParams(recurrent_gain=2.0))
        brain.activity[:] = [0.3, 0.6, 0.2, 0.4]

        parts = diagnostic.recurrent_components(
            brain,
            np.array([3]),
            {"heading": np.array([0]), "goal": np.array([1])},
        )

        self.assertAlmostEqual(float(parts["heading"][0]), 0.2)
        self.assertAlmostEqual(float(parts["goal"][0]), -0.2)
        self.assertAlmostEqual(float(parts["other"][0]), 0.2)
        expected = brain.params.recurrent_gain * (brain.W[[3]] @ brain.activity)
        np.testing.assert_allclose(sum(parts.values()), expected)

    def test_first_harmonic_recovers_preferred_phase(self):
        diagnostic = importlib.import_module("diagnose_stage1")
        self.assertTrue(
            hasattr(diagnostic, "fit_first_harmonic"),
            "fit_first_harmonic is missing",
        )
        angles = np.linspace(-math.pi, math.pi, 16, endpoint=False)
        preferred = np.array([-1.2, 0.7])
        responses = 0.3 + 0.5 * np.cos(angles[:, None] - preferred)

        fitted = diagnostic.fit_first_harmonic(angles, responses)

        np.testing.assert_allclose(fitted["phase"], preferred, atol=1e-12)
        np.testing.assert_allclose(fitted["amplitude"], 0.5, atol=1e-12)
        np.testing.assert_allclose(fitted["r2"], 1.0, atol=1e-12)

    def test_alignment_recovers_handedness_and_offset(self):
        diagnostic = importlib.import_module("diagnose_stage1")
        self.assertTrue(
            hasattr(diagnostic, "best_circular_alignment"),
            "best_circular_alignment is missing",
        )
        anatomical = np.linspace(-math.pi, math.pi, 8, endpoint=False)
        measured = np.angle(np.exp(1j * (-anatomical + 0.4)))

        fitted = diagnostic.best_circular_alignment(measured, anatomical)

        self.assertEqual(fitted["handedness"], -1)
        self.assertAlmostEqual(fitted["offset"], 0.4)
        self.assertAlmostEqual(fitted["concentration"], 1.0)

if __name__ == "__main__":
    unittest.main()
