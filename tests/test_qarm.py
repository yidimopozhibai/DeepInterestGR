from __future__ import annotations

import math
import unittest

from deepinterestgr import (
    ADVANTAGE_EPSILON,
    ALPHA,
    SIDCatalog,
    SemanticID,
    aggregate_quality,
    clipped_grpo_surrogate,
    normalized_advantages,
    reward_breakdown,
)


class QARMEquationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = SIDCatalog(
            {
                "target": SemanticID(("a1", "b2", "c3")),
                "wrong-high-quality": SemanticID(("a1", "b9", "c9")),
                "collision-a": SemanticID(("a7", "b7", "c7"), "u1"),
                "collision-b": SemanticID(("a7", "b7", "c7"), "u2"),
            }
        )

    def test_quality_aggregation_equation_and_missing_fallback(self):
        self.assertAlmostEqual(aggregate_quality([1, 0, 1], [0.6, 0.3, 0.1]), 0.7)
        self.assertAlmostEqual(aggregate_quality([1, 0, 1], [None, None, None]), 2 / 3)
        with self.assertRaisesRegex(ValueError, "all present or all missing"):
            aggregate_quality([1, 0], [0.8, None])
        with self.assertRaisesRegex(ValueError, "denominator"):
            aggregate_quality([1, 0], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregate_quality([1], [float("nan")])
        with self.assertRaisesRegex(ValueError, "integer"):
            aggregate_quality([True], [1.0])

    def test_collision_suffix_decodes_identity_but_is_excluded_from_lcp(self):
        audit = reward_breakdown(
            candidate=("a7", "b7", "c7", "u2"),
            target_item_id="collision-a",
            catalog=self.catalog,
            item_quality={"collision-a": 1.0},
            lambda_valid=0.2,
            lambda_pref=0.3,
        )
        self.assertEqual(audit.decoded_item_id, "collision-b")
        self.assertEqual(audit.lcp_length, 3)
        self.assertEqual(audit.lcp, 0.3)
        self.assertEqual(audit.exact, 0.0)
        self.assertEqual(audit.quality, 0.0)
        self.assertAlmostEqual(audit.total, 0.5)

    def test_catalog_enforces_collision_suffix_rules(self):
        with self.assertRaisesRegex(ValueError, "needs a collision suffix"):
            SIDCatalog(
                {
                    "a": SemanticID(("a", "b", "c")),
                    "b": SemanticID(("a", "b", "c"), "u2"),
                }
            )
        with self.assertRaisesRegex(ValueError, "forbidden"):
            SIDCatalog({"a": SemanticID(("a", "b", "c"), "unneeded")})
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            SemanticID(("a", "b"))
        with self.assertRaisesRegex(ValueError, "at most one suffix"):
            SemanticID.from_tokens(("a", "b", "c", "u1", "extra"))
        with self.assertRaisesRegex(ValueError, "sequence of token strings"):
            SemanticID.from_tokens("abc")

    def test_malformed_sid_is_invalid_without_aborting_reward(self):
        short = reward_breakdown(
            candidate=("a1", "b2"),
            target_item_id="target",
            catalog=self.catalog,
            item_quality={"target": 1.0},
            lambda_valid=0.2,
            lambda_pref=0.3,
        )
        self.assertFalse(short.candidate_valid)
        self.assertIsNone(short.decoded_item_id)
        self.assertEqual(short.lcp_length, 2)
        self.assertEqual(short.exact, 0.0)
        self.assertEqual(short.valid, 0.0)
        self.assertEqual(short.quality, 0.0)
        self.assertAlmostEqual(short.total, 0.3 * 2 / 3)

        overlong = reward_breakdown(
            candidate=("a1", "b2", "c3", "suffix", "extra"),
            target_item_id="target",
            catalog=self.catalog,
            item_quality={"target": 1.0},
            lambda_valid=0.2,
            lambda_pref=0.3,
        )
        self.assertFalse(overlong.candidate_valid)
        self.assertEqual(overlong.lcp_length, 3)
        self.assertAlmostEqual(overlong.total, 0.3)

    def test_reward_equation_and_wrong_high_quality_item_gets_zero_quality(self):
        exact = reward_breakdown(
            candidate=("a1", "b2", "c3"),
            target_item_id="target",
            catalog=self.catalog,
            item_quality={"target": 0.8, "wrong-high-quality": 1.0},
            lambda_valid=0.2,
            lambda_pref=0.3,
        )
        self.assertEqual(ALPHA, 0.5)
        self.assertEqual((exact.exact, exact.valid, exact.lcp, exact.quality), (1.0, 0.2, 0.3, 0.8))
        self.assertAlmostEqual(exact.total, 1.0 + 0.2 + 0.3 + 0.5 * 0.8)

        wrong = reward_breakdown(
            candidate=("a1", "b9", "c9"),
            target_item_id="target",
            catalog=self.catalog,
            item_quality={"target": 0.0, "wrong-high-quality": 1.0},
            lambda_valid=0.2,
            lambda_pref=0.3,
        )
        self.assertEqual(wrong.decoded_item_id, "wrong-high-quality")
        self.assertEqual(wrong.quality, 0.0)
        self.assertEqual(wrong.exact, 0.0)
        self.assertAlmostEqual(wrong.total, 0.2 + 0.3 / 3)

    def test_lambda_values_are_caller_required(self):
        with self.assertRaises(TypeError):
            reward_breakdown(
                candidate=("a1", "b2", "c3"),
                target_item_id="target",
                catalog=self.catalog,
                item_quality={"target": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            reward_breakdown(
                candidate=("a1", "b2", "c3"),
                target_item_id="target",
                catalog=self.catalog,
                item_quality={"target": 1.0},
                lambda_valid=-0.1,
                lambda_pref=0.2,
            )

    def test_advantage_equation_uses_population_std_and_epsilon(self):
        advantages = normalized_advantages((1.0, 2.0, 3.0))
        expected_scale = math.sqrt(2.0 / 3.0) + 1e-6
        self.assertEqual(ADVANTAGE_EPSILON, 1e-6)
        self.assertAlmostEqual(advantages[0], -1.0 / expected_scale)
        self.assertAlmostEqual(advantages[1], 0.0)
        self.assertAlmostEqual(advantages[2], 1.0 / expected_scale)
        self.assertEqual(normalized_advantages((4.0, 4.0)), (0.0, 0.0))

    def test_clipped_grpo_surrogate_positive_negative_and_kl(self):
        result = clipped_grpo_surrogate(
            new_log_probs=(math.log(1.5), math.log(0.5)),
            behavior_log_probs=(0.0, 0.0),
            advantages=(2.0, -2.0),
            kl_divergence=0.4,
            clip_epsilon=0.2,
            kl_beta=0.5,
        )
        self.assertEqual(result.clipped_ratios, (1.2, 0.8))
        # min(1.5*2, 1.2*2)=2.4; min(0.5*-2, 0.8*-2)=-1.6
        self.assertAlmostEqual(result.policy_mean, 0.4)
        self.assertAlmostEqual(result.kl_penalty, 0.2)
        self.assertAlmostEqual(result.objective, 0.2)
        with self.assertRaisesRegex(ValueError, "likelihood ratios"):
            clipped_grpo_surrogate(
                new_log_probs=(1000.0,),
                behavior_log_probs=(0.0,),
                advantages=(1.0,),
                kl_divergence=0.0,
                clip_epsilon=0.2,
                kl_beta=0.0,
            )

    def test_clip_epsilon_and_kl_beta_are_required(self):
        with self.assertRaises(TypeError):
            clipped_grpo_surrogate(
                new_log_probs=(0.0,),
                behavior_log_probs=(0.0,),
                advantages=(1.0,),
                kl_divergence=0.0,
            )


if __name__ == "__main__":
    unittest.main()
