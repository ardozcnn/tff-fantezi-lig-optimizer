import unittest

import pandas as pd

from calibrate_leagues import is_domestic_league, season_start
from src.fetch_fotmob import readiness_multiplier
from src.league_translation import (
    translate_external_rates,
    translate_metric,
    translate_metric_mixture,
)
from src.scoring import (
    _blend_rate_sets,
    _empty_rates,
    _recency_multiplier,
    apply_context_adjustments,
    expected_points_from_rates,
    lookup_fixture_context,
    recency_for_projection,
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

    def test_leftover_full_season_tff_points_are_ignored(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "projected_pts": 4.0,
                    "price_m": 10.0,
                    "data_src": "super_lig",
                    "form_apps": 1,
                    "availability": "AVAILABLE",
                    "tff_minutes": 1886,
                    "tff_starts": 25,
                    "tff_points": 150,
                    "tff_ppm": 6.0,
                }
            ]
        )

        adjusted = apply_context_adjustments(frame)

        self.assertEqual(adjusted.loc[0, "projected_pts"], 4.0)
        self.assertEqual(adjusted.loc[0, "tff_calibration_weight"], 0.0)

    def test_fixture_team_aliases_match(self) -> None:
        context = {"basaksehir fk": {"opponent": "Kocaelispor"}}
        fixture = lookup_fixture_context("İstanbul Başakşehir", context)
        self.assertEqual(fixture["opponent"], "Kocaelispor")

    def test_missing_recent_appearances_reduce_weekly_projection(self) -> None:
        self.assertEqual(_recency_multiplier(6, 6), 1.0)
        self.assertAlmostEqual(_recency_multiplier(0, 6), 0.65)
        self.assertGreater(_recency_multiplier(5, 6), 0.9)
        self.assertEqual(recency_for_projection(1, 6, preseason=True), 1.0)
        self.assertAlmostEqual(recency_for_projection(1, 6, preseason=False), 0.65 + 0.35 / 6)

    def test_current_club_matches_restore_readiness(self) -> None:
        self.assertEqual(readiness_multiplier(4, 280), 1.0)
        self.assertGreater(readiness_multiplier(2, 180), 0.85)
        self.assertEqual(readiness_multiplier(0, 0), 0.65)

    def test_exact_league_translation_beats_global_fallback(self) -> None:
        model = {
            "global": {
                "positions": {
                    "MF": {
                        "goals_p90": {
                            "intercept": 0.0,
                            "slope": 0.9,
                            "cap": 2.0,
                            "n_players": 100,
                        }
                    }
                }
            },
            "leagues": {
                "98": {
                    "name": "Trendyol 1.Lig",
                    "positions": {
                        "MF": {
                            "goals_p90": {
                                "intercept": 0.0,
                                "slope": 0.7,
                                "cap": 2.0,
                                "n_players": 25,
                                "local_weight": 0.6,
                            }
                        }
                    },
                }
            },
        }
        exact, exact_meta = translate_metric(model, 98, "MF", "goals_p90", 0.5)
        fallback, fallback_meta = translate_metric(model, 999, "MF", "goals_p90", 0.5)
        mixed, _ = translate_metric_mixture(
            model,
            [
                {"tournament_id": 98, "weight": 0.5},
                {"tournament_id": 999, "weight": 0.5},
            ],
            "MF",
            "goals_p90",
            0.5,
        )

        self.assertAlmostEqual(exact, 0.35)
        self.assertAlmostEqual(fallback, 0.45)
        self.assertAlmostEqual(mixed, 0.40)
        self.assertEqual(exact_meta["level"], "league")
        self.assertEqual(fallback_meta["level"], "global")

    def test_star_goal_rate_mixes_toward_source_not_squad_mean(self) -> None:
        model = {
            "leagues": {
                "34": {
                    "name": "Ligue 1",
                    "observed_source_ga_p90": 0.20,
                    "positions": {
                        "FW": {
                            "goals_p90": {
                                "intercept": 0.22,
                                "slope": 0.55,
                                "cap": 1.22,
                                "n_players": 13,
                                "local_weight": 0.7,
                            }
                        }
                    },
                }
            }
        }
        source = 0.61
        predicted, meta = translate_metric(model, 34, "FW", "goals_p90", source)
        regression = 0.22 + 0.55 * source

        self.assertGreater(predicted, regression)
        self.assertLess(predicted, source)
        self.assertGreater(meta["identity_mix"], 0.4)

    def test_typical_goal_rate_keeps_regression(self) -> None:
        model = {
            "leagues": {
                "34": {
                    "name": "Ligue 1",
                    "observed_source_ga_p90": 0.20,
                    "positions": {
                        "FW": {
                            "goals_p90": {
                                "intercept": 0.0,
                                "slope": 0.66,
                                "cap": 1.22,
                                "n_players": 13,
                                "local_weight": 0.7,
                            }
                        }
                    },
                }
            }
        }
        source = 0.10
        predicted, meta = translate_metric(model, 34, "FW", "goals_p90", source)
        self.assertAlmostEqual(predicted, 0.066)
        self.assertEqual(meta["identity_mix"], 0.0)

    def test_tier_peer_fallback_beats_global_when_league_weight_is_zero(self) -> None:
        model = {
            "global": {
                "positions": {
                    "FW": {
                        "goals_p90": {
                            "intercept": 0.0,
                            "slope": 0.4,
                            "cap": 2.0,
                            "n_players": 100,
                        }
                    }
                }
            },
            "leagues": {
                "34": {
                    "name": "Ligue 1",
                    "positions": {
                        "FW": {
                            "goals_p90": {
                                "intercept": 0.0,
                                "slope": 0.8,
                                "cap": 2.0,
                                "n_players": 20,
                                "local_weight": 0.6,
                            }
                        }
                    },
                },
                "17": {"name": "Premier League", "positions": {}},
            },
        }
        predicted, meta = translate_metric(model, 17, "FW", "goals_p90", 0.5)
        self.assertEqual(meta["level"], "tier")
        self.assertAlmostEqual(predicted, 0.4)

    def test_translation_converts_per90_back_to_projected_minutes(self) -> None:
        model = {
            "global": {
                "positions": {
                    "MF": {
                        "goals_p90": {
                            "intercept": 0.0,
                            "slope": 0.5,
                            "cap": 2.0,
                            "n_players": 50,
                        },
                        "assists_p90": {
                            "intercept": 0.0,
                            "slope": 1.0,
                            "cap": 2.0,
                            "n_players": 50,
                        },
                        "clean_sheet_rate": {
                            "intercept": 0.0,
                            "slope": 1.0,
                            "cap": 1.0,
                            "n_players": 50,
                        },
                    }
                }
            },
            "leagues": {},
        }
        rates = _empty_rates()
        rates.update(
            {
                "apps": 10.0,
                "min_per_app": 90.0,
                "gls_pa": 0.4,
                "ast_pa": 0.2,
                "cs_rate": 0.3,
            }
        )

        translated, _ = translate_external_rates(
            rates,
            {"tournament_id": 999, "tournament": "Test League"},
            "MF",
            60.0,
            model=model,
        )

        self.assertAlmostEqual(translated["gls_pa"], 0.4 * 0.5 * 60 / 90)
        self.assertAlmostEqual(translated["ast_pa"], 0.2 * 60 / 90)

    def test_team_clean_sheet_prior_overrides_old_club_conceding_rate(self) -> None:
        low_concede = _empty_rates()
        low_concede.update(
            {
                "apps": 10.0,
                "min_per_app": 90.0,
                "share_60": 1.0,
                "ga_pa": 0.4,
                "cs_rate": 0.5,
            }
        )
        high_concede = low_concede.copy()
        high_concede["ga_pa"] = 2.4

        low = expected_points_from_rates(
            low_concede, "GK", team_cs_rate=0.45, appearance=1.0
        )
        high = expected_points_from_rates(
            high_concede, "GK", team_cs_rate=0.45, appearance=1.0
        )

        self.assertAlmostEqual(low, high)

    def test_calibration_source_filter_keeps_leagues_not_cups(self) -> None:
        self.assertEqual(season_start("24/25"), 2024)
        self.assertTrue(
            is_domestic_league(
                {"tournament_id": 98, "tournament": "Trendyol 1.Lig"}
            )
        )
        self.assertTrue(
            is_domestic_league(
                {"tournament_id": 53, "tournament": "Serie B"}
            )
        )
        self.assertFalse(
            is_domestic_league(
                {"tournament_id": 96, "tournament": "Türkiye Kupası"}
            )
        )
        self.assertFalse(
            is_domestic_league(
                {"tournament_id": 373, "tournament": "Copa Betano do Brasil"}
            )
        )


if __name__ == "__main__":
    unittest.main()
