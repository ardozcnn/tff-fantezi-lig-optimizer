import unittest

import pandas as pd

from calibrate_leagues import is_domestic_league, season_start
from src.fetch_fotmob import (
    hot_form_blend_weight,
    hot_form_expected_points,
    readiness_multiplier,
    _soften_rate,
)
from src.manager_cards import choose_manager_card, manager_card_advice
from src.league_translation import (
    translate_external_rates,
    translate_metric,
    translate_metric_mixture,
)
from src.fetch_stats import blend_team_cs_rates
from src.scoring import (
    _blend_rate_sets,
    _empty_rates,
    _recency_multiplier,
    apply_context_adjustments,
    blend_weights,
    expected_points_from_rates,
    lookup_fixture_context,
    recency_for_projection,
    shrink_small_sample_rates,
)


class ScoringTests(unittest.TestCase):
    def test_manager_cards_score_captain_and_full_bench_uplift(self) -> None:
        xi = pd.DataFrame(
            [
                {"player": "Kaleci", "team": "A", "position": "GK", "price_m": 4.0, "projected_pts": 3.0},
                {"player": "Defans 1", "team": "B", "position": "DF", "price_m": 4.0, "projected_pts": 3.0},
                {"player": "Defans 2", "team": "C", "position": "DF", "price_m": 4.0, "projected_pts": 3.0},
                {"player": "Defans 3", "team": "D", "position": "DF", "price_m": 4.0, "projected_pts": 3.0},
                {"player": "Defans 4", "team": "E", "position": "DF", "price_m": 4.0, "projected_pts": 3.0},
                {"player": "Orta 1", "team": "F", "position": "MF", "price_m": 5.0, "projected_pts": 3.0},
                {"player": "Orta 2", "team": "G", "position": "MF", "price_m": 5.0, "projected_pts": 3.0},
                {"player": "Orta 3", "team": "H", "position": "MF", "price_m": 5.0, "projected_pts": 3.0},
                {"player": "Orta 4", "team": "I", "position": "MF", "price_m": 5.0, "projected_pts": 3.0},
                {"player": "Forvet 1", "team": "J", "position": "FW", "price_m": 6.0, "projected_pts": 3.0},
                {"player": "Kaptan", "team": "K", "position": "FW", "price_m": 6.0, "projected_pts": 5.0},
            ]
        )
        bench = pd.DataFrame(
            [
                {"player": "Yedek 1", "projected_pts": 4.0},
                {"player": "Yedek 2", "projected_pts": 4.0},
            ]
        )
        pool = pd.concat([xi, bench], ignore_index=True).fillna(
            {"team": "Y", "position": "MF", "price_m": 4.0}
        )
        result = {
            "xi": xi,
            "bench": bench,
            "captain": {"player": "Kaptan", "projected_pts": 5.0},
            "total_projected": 38.0,
        }
        cards = {card["card"]: card for card in manager_card_advice(result, pool, budget=100)}

        self.assertEqual(cards["Tripleks Kaptan"]["extra_pts"], 5.0)
        self.assertEqual(cards["Dört Dörtlük Kaptan"]["extra_pts"], 10.0)
        self.assertEqual(cards["Tüm Takım Sahaya"]["extra_pts"], 4.0)

    def test_manager_recommends_only_one_card_or_hold(self) -> None:
        hold = choose_manager_card(
            [
                {"card": "Dört Dörtlük Kaptan", "extra_pts": 10.9, "why": "4x"},
                {"card": "Tüm Takım Sahaya", "extra_pts": 7.4, "why": "bench"},
            ]
        )
        use = choose_manager_card(
            [
                {"card": "Tripleks Kaptan", "extra_pts": 6.4, "why": "3x"},
                {"card": "Tüm Takım Sahaya", "extra_pts": 7.0, "why": "bench"},
            ]
        )

        self.assertFalse(hold["use"])
        self.assertEqual(hold["card"], "Kart kullanma")
        self.assertTrue(use["use"])
        self.assertEqual(use["card"], "Tripleks Kaptan")

    def test_fotmob_hot_form_softens_two_goal_burst(self) -> None:
        self.assertAlmostEqual(
            _soften_rate(2.0, 1.0, 0.40, 6.0),
            4.4 / 7.0,
            places=5,
        )
        first_week_weight = hot_form_blend_weight(
            1.0,
            90.0,
            early_season=True,
        )
        self.assertGreater(first_week_weight, 0.07)
        self.assertLess(first_week_weight, 0.10)
        first_form, first_base = blend_weights(1.0)
        full_form, full_base = blend_weights(6.0)
        self.assertAlmostEqual(first_form, 0.05)
        self.assertAlmostEqual(first_base, 0.95)
        self.assertAlmostEqual(full_form, 0.30)
        self.assertAlmostEqual(full_base, 0.70)

        hot = hot_form_expected_points(
            {
                "fotmob_sl_apps": 1.0,
                "fotmob_sl_minutes": 90.0,
                "fotmob_sl_goals": 2.0,
                "fotmob_sl_assists": 0.0,
                "fotmob_sl_rating": 9.0,
            },
            "FW",
        )
        self.assertIsNotNone(hot)
        assert hot is not None
        self.assertGreater(hot, 4.0)
        self.assertLess(hot, 7.0)

        blended = (1.0 - first_week_weight) * 4.97 + first_week_weight * hot
        self.assertGreater(blended, 4.8)
        self.assertLess(blended, 5.2)

    def test_current_small_sample_is_blended_not_discarded(self) -> None:
        current = _empty_rates()
        current.update({"apps": 7.0, "gls_pa": 2 / 7, "ast_pa": 0.0})
        previous = _empty_rates()
        previous.update({"apps": 34.0, "gls_pa": 9 / 34, "ast_pa": 8 / 34})

        blended, current_weight = _blend_rate_sets(current, previous, 7.0)

        self.assertAlmostEqual(current_weight, 7 / 17)
        self.assertGreater(blended["gls_pa"], previous["gls_pa"])
        self.assertLess(blended["ast_pa"], previous["ast_pa"])

    def test_single_clean_sheet_is_shrunk_for_goalkeepers(self) -> None:
        one_match = _empty_rates()
        one_match.update(
            {
                "apps": 1.0,
                "apps_60": 1.0,
                "share_60": 1.0,
                "min_per_app": 90.0,
                "cs_rate": 1.0,
                "saves_pa": 1.0,
                "ga_pa": 0.0,
                "rating": 7.4,
            }
        )
        shrunk, weight = shrink_small_sample_rates(one_match, "GK")
        raw = expected_points_from_rates(one_match, "GK", appearance=1.0)
        tempered = expected_points_from_rates(shrunk, "GK", appearance=1.0)

        self.assertAlmostEqual(weight, 1 / 9)
        self.assertLess(shrunk["cs_rate"], 0.45)
        self.assertLess(tempered, raw)
        self.assertLess(tempered, 5.0)

    def test_early_team_cs_is_blended_not_taken_as_certainty(self) -> None:
        blended = blend_team_cs_rates(
            {"Rizespor": 1.0, "Başakşehir": 0.0},
            {"Rizespor": 0.30, "Başakşehir": 0.36},
            {"Rizespor": 1.0, "Başakşehir": 0.0},
        )
        self.assertGreater(blended["Rizespor"], 0.30)
        self.assertLess(blended["Rizespor"], 0.45)
        self.assertAlmostEqual(blended["Başakşehir"], 0.36)

        perfect = expected_points_from_rates(
            {"apps": 10, "share_60": 1.0, "min_per_app": 90, "saves_pa": 3.0},
            "GK",
            team_cs_rate=1.0,
            appearance=1.0,
        )
        tempered = expected_points_from_rates(
            {"apps": 10, "share_60": 1.0, "min_per_app": 90, "saves_pa": 3.0},
            "GK",
            team_cs_rate=blended["Rizespor"],
            appearance=1.0,
        )
        self.assertLess(tempered, perfect - 1.5)

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

    def test_first_official_match_has_conservative_weight(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "projected_pts": 5.0,
                    "form_apps": 1,
                    "availability": "AVAILABLE",
                    "tff_minutes": 90,
                    "tff_starts": 1,
                    "tff_points": 13,
                    "tff_ppm": 13,
                }
            ]
        )

        adjusted = apply_context_adjustments(frame)

        weight = float(adjusted.loc[0, "tff_calibration_weight"])
        self.assertGreater(weight, 0.08)
        self.assertLess(weight, 0.10)
        self.assertLess(float(adjusted.loc[0, "projected_pts"]), 6.0)

    def test_goalkeeper_depth_prevents_single_old_clean_sheet_from_starting(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "player": "Takımın Birincisi",
                    "team": "Başakşehir",
                    "position": "GK",
                    "projected_pts": 4.8,
                    "base_apps": 33,
                    "min_per_app": 90,
                    "form_apps": 0,
                    "availability": "AVAILABLE",
                },
                {
                    "player": "Tek Maçlık Yedek",
                    "team": "Başakşehir",
                    "position": "GK",
                    "projected_pts": 6.4,
                    "base_apps": 1,
                    "min_per_app": 90,
                    "form_apps": 0,
                    "availability": "AVAILABLE",
                },
            ]
        )

        adjusted = apply_context_adjustments(frame)

        self.assertAlmostEqual(adjusted.loc[0, "gk_start_probability"], 0.95)
        self.assertLess(adjusted.loc[1, "gk_start_probability"], 0.10)
        self.assertGreater(
            adjusted.loc[0, "projected_pts"],
            adjusted.loc[1, "projected_pts"],
        )

    def test_injured_and_suspended_players_are_not_selection_eligible(self) -> None:
        frame = pd.DataFrame(
            [
                {"projected_pts": 5.0, "availability": "AVAILABLE"},
                {"projected_pts": 5.0, "availability": "INJURED"},
                {"projected_pts": 5.0, "availability": "SUSPENDED"},
                {"projected_pts": 5.0, "availability": "DOUBTFUL", "avail_pct": 60},
            ]
        )

        adjusted = apply_context_adjustments(frame)

        self.assertEqual(adjusted["selection_eligible"].tolist(), [True, False, False, True])
        self.assertLess(adjusted.loc[1, "projected_pts"], 5.0)
        self.assertLess(adjusted.loc[2, "projected_pts"], 5.0)

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
