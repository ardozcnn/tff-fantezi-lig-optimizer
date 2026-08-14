import unittest

import pandas as pd

from src.fetch_fotmob import readiness_multiplier
from src.scoring import (
    _blend_rate_sets,
    _empty_rates,
    _recency_multiplier,
    apply_context_adjustments,
    expected_points_from_rates,
    lookup_fixture_context,
)


class ScoringTests(unittest.TestCase):
    def test_current_small_sample_is_blended_not_discarded(self) -> None:
        current = _empty_rates()
        current.update({"apps": 7.0, "gls_pa": 2 / 7, "ast_pa": 0.0})
        previous = _empty_rates()
        previous.update({"apps": 34.0, "gls_pa": 9 / 34, "ast_pa": 8 / 34})

        blended, current_weight = _blend_rate_sets(current, previous, 7.0)

        self.assertAlmostEqual(current_weight, 7 / 12)
        self.assertGreater(blended["gls_pa"], previous["gls_pa"])
        self.assertLess(blended["ast_pa"], previous["ast_pa"])

    def test_appearance_probability_scales_attacking_returns(self) -> None:
        rates = _empty_rates()
        rates.update(
            {
                "apps": 10.0,
                "min_per_app": 80.0,
                "share_60": 1.0,
                "gls_pa": 0.5,
                "xg_pa": 0.5,
                "ast_pa": 0.2,
                "xa_pa": 0.2,
                "rating": 7.0,
            }
        )

        certain = expected_points_from_rates(rates, "MF", appearance=1.0)
        half = expected_points_from_rates(rates, "MF", appearance=0.5)

        self.assertAlmostEqual(half, certain * 0.5, places=6)

    def test_cards_reduce_projection(self) -> None:
        clean = _empty_rates()
        clean.update({"apps": 10.0, "min_per_app": 80.0, "share_60": 1.0})
        carded = clean.copy()
        carded.update({"yc_pa": 0.2, "rc_pa": 0.1})

        clean_points = expected_points_from_rates(clean, "MF", appearance=1.0)
        carded_points = expected_points_from_rates(carded, "MF", appearance=1.0)

        self.assertAlmostEqual(clean_points - carded_points, 0.5, places=6)

    def test_official_tff_points_calibrate_only_after_minutes_exist(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "projected_pts": 4.0,
                    "price_m": 7.0,
                    "data_src": "super_lig",
                    "form_apps": 6,
                    "availability": "AVAILABLE",
                    "tff_minutes": 750,
                    "tff_starts": 10,
                    "tff_points": 60,
                    "tff_ppm": 6.0,
                },
                {
                    "projected_pts": 4.0,
                    "price_m": 7.0,
                    "data_src": "super_lig",
                    "form_apps": 6,
                    "availability": "AVAILABLE",
                    "tff_minutes": 0,
                    "tff_starts": 0,
                    "tff_points": 0,
                    "tff_ppm": 0,
                },
            ]
        )

        adjusted = apply_context_adjustments(frame)

        self.assertGreater(adjusted.loc[0, "projected_pts"], 4.0)
        self.assertEqual(adjusted.loc[1, "projected_pts"], 4.0)

    def test_fixture_team_aliases_match(self) -> None:
        context = {"basaksehir fk": {"opponent": "Kocaelispor"}}
        fixture = lookup_fixture_context("İstanbul Başakşehir", context)
        self.assertEqual(fixture["opponent"], "Kocaelispor")

    def test_missing_recent_appearances_reduce_weekly_projection(self) -> None:
        self.assertEqual(_recency_multiplier(6, 6), 1.0)
        self.assertAlmostEqual(_recency_multiplier(0, 6), 0.65)
        self.assertGreater(_recency_multiplier(5, 6), 0.9)

    def test_current_club_matches_restore_readiness(self) -> None:
        self.assertEqual(readiness_multiplier(4, 280), 1.0)
        self.assertGreater(readiness_multiplier(2, 180), 0.85)
        self.assertEqual(readiness_multiplier(0, 0), 0.65)


if __name__ == "__main__":
    unittest.main()
