import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
                {"player": "Kaleci", "team": "A", "position": "GK", "price_m": 4.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Defans 1", "team": "B", "position": "DF", "price_m": 4.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Defans 2", "team": "C", "position": "DF", "price_m": 4.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Defans 3", "team": "D", "position": "DF", "price_m": 4.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Defans 4", "team": "E", "position": "DF", "price_m": 4.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Orta 1", "team": "F", "position": "MF", "price_m": 5.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Orta 2", "team": "G", "position": "MF", "price_m": 5.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Orta 3", "team": "H", "position": "MF", "price_m": 5.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Orta 4", "team": "I", "position": "MF", "price_m": 5.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Forvet 1", "team": "J", "position": "FW", "price_m": 6.0, "projected_pts": 3.0, "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "Kaptan", "team": "K", "position": "FW", "price_m": 6.0, "projected_pts": 5.0, "pts_if_plays": 5.0, "play_probability": 1.0},
            ]
        )
        bench = pd.DataFrame(
            [
                {"player": "Yedek 1", "position": "MF", "projected_pts": 4.0, "pts_if_plays": 4.0, "play_probability": 1.0},
                {"player": "Yedek 2", "position": "FW", "projected_pts": 4.0, "pts_if_plays": 4.0, "play_probability": 1.0},
            ]
        )
        pool = pd.concat([xi, bench], ignore_index=True).fillna(
            {"team": "Y", "position": "MF", "price_m": 4.0}
        )
        result = {
            "xi": xi,
            "bench": bench,
            "captain": {"player": "Kaptan", "projected_pts": 5.0, "pts_if_plays": 5.0, "play_probability": 1.0},
            "total_projected": 38.0,
        }
        cards = {card["card"]: card for card in manager_card_advice(result, pool, budget=100)}

        self.assertEqual(cards["Tripleks Kaptan"]["extra_pts"], 5.0)
        self.assertEqual(cards["Dört Dörtlük Kaptan"]["extra_pts"], 10.0)
        self.assertGreaterEqual(cards["Tüm Takım Sahaya"]["extra_pts"], 0.0)

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
        self.assertAlmostEqual(first_form, 1.0 / 5.0)
        self.assertAlmostEqual(first_base, 4.0 / 5.0)
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
        self.assertGreater(shrunk["saves_pa"], 2.0)
        self.assertLess(tempered, raw)
        self.assertLess(tempered, 5.0)

    def test_count_metrics_get_more_early_season_weight_than_rare_events(self) -> None:
        current = _empty_rates()
        current.update(
            {
                "apps": 1.0,
                "cs_rate": 1.0,
                "saves_pa": 6.0,
                "gls_pa": 0.0,
            }
        )
        previous = _empty_rates()
        previous.update(
            {
                "apps": 30.0,
                "cs_rate": 0.30,
                "saves_pa": 3.0,
                "gls_pa": 0.0,
            }
        )
        blended, _ = _blend_rate_sets(current, previous, 1.0)
        self.assertLess(blended["cs_rate"], 0.45)
        self.assertGreater(blended["saves_pa"], 3.5)
        self.assertGreater(blended["saves_pa"], blended["cs_rate"] + 3.0)

    def test_early_team_cs_is_blended_not_taken_as_certainty(self) -> None:
        blended = blend_team_cs_rates(
            {"Rizespor": 1.0, "Başakşehir": 0.0},
            {"Rizespor": 0.30, "Başakşehir": 0.36},
            {"Rizespor": 1.0, "Başakşehir": 0.0},
        )
        self.assertGreater(blended["Rizespor"], 0.30)
        self.assertLess(blended["Rizespor"], 0.45)
        self.assertAlmostEqual(blended["Başakşehir"], 0.36)

        four = blend_team_cs_rates(
            {"Rizespor": 1.0},
            {"Rizespor": 0.30},
            {"Rizespor": 4.0},
        )
        self.assertLess(four["Rizespor"], 1.0)
        self.assertAlmostEqual(four["Rizespor"], 4 / 12 * 1.0 + 8 / 12 * 0.30, places=5)

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
                    "player": "A",
                    "team": "X",
                    "position": "MF",
                    "projected_pts": 4.0,
                    "price_m": 7.0,
                    "data_src": "super_lig",
                    "form_apps": 6,
                    "min_per_app": 90,
                    "availability": "AVAILABLE",
                    "tff_minutes": 750,
                    "tff_starts": 10,
                    "tff_points": 60,
                    "tff_ppm": 6.0,
                },
                {
                    "player": "B",
                    "team": "Y",
                    "position": "MF",
                    "projected_pts": 4.0,
                    "price_m": 7.0,
                    "data_src": "super_lig",
                    "form_apps": 6,
                    "min_per_app": 90,
                    "availability": "AVAILABLE",
                    "tff_minutes": 0,
                    "tff_starts": 0,
                    "tff_points": 0,
                    "tff_ppm": 0,
                },
            ]
        )

        adjusted = apply_context_adjustments(frame)

        self.assertGreater(adjusted.loc[0, "pts_if_plays"], 4.0)
        self.assertAlmostEqual(adjusted.loc[1, "pts_if_plays"], 4.0)

    def test_first_official_match_has_conservative_weight(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "player": "Osimhen",
                    "team": "Galatasaray",
                    "position": "FW",
                    "projected_pts": 5.0,
                    "form_apps": 1,
                    "min_per_app": 90,
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
        self.assertGreater(weight, 0.12)
        self.assertLess(weight, 0.20)
        self.assertGreater(float(adjusted.loc[0, "pts_if_plays"]), 5.0)
        self.assertLess(float(adjusted.loc[0, "pts_if_plays"]), 7.0)

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
                {
                    "player": "A",
                    "team": "T",
                    "position": "MF",
                    "projected_pts": 5.0,
                    "min_per_app": 90,
                    "form_apps": 6,
                    "availability": "AVAILABLE",
                },
                {
                    "player": "B",
                    "team": "T",
                    "position": "MF",
                    "projected_pts": 5.0,
                    "min_per_app": 90,
                    "form_apps": 6,
                    "availability": "INJURED",
                },
                {
                    "player": "C",
                    "team": "T",
                    "position": "MF",
                    "projected_pts": 5.0,
                    "min_per_app": 90,
                    "form_apps": 6,
                    "availability": "SUSPENDED",
                },
                {
                    "player": "D",
                    "team": "T",
                    "position": "MF",
                    "projected_pts": 5.0,
                    "min_per_app": 90,
                    "form_apps": 6,
                    "availability": "DOUBTFUL",
                    "avail_pct": 60,
                },
            ]
        )

        adjusted = apply_context_adjustments(frame)

        self.assertEqual(adjusted["selection_eligible"].tolist(), [True, False, False, True])
        self.assertEqual(adjusted.loc[1, "play_probability"], 0.0)
        self.assertEqual(adjusted.loc[2, "play_probability"], 0.0)
        self.assertLess(adjusted.loc[3, "play_probability"], adjusted.loc[0, "play_probability"])

    def test_leftover_full_season_tff_points_are_ignored(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "player": "Asensio",
                    "team": "Fenerbahçe",
                    "position": "MF",
                    "projected_pts": 4.0,
                    "price_m": 10.0,
                    "data_src": "super_lig",
                    "form_apps": 1,
                    "min_per_app": 90,
                    "availability": "AVAILABLE",
                    "tff_minutes": 1886,
                    "tff_starts": 25,
                    "tff_points": 150,
                    "tff_ppm": 6.0,
                }
            ]
        )

        adjusted = apply_context_adjustments(frame)

        self.assertAlmostEqual(adjusted.loc[0, "pts_if_plays"], 4.0)
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

    def test_autosub_replaces_goalkeeper_only_with_goalkeeper(self) -> None:
        from src.autosub import apply_autosub, is_legal_xi

        self.assertTrue(is_legal_xi(["GK", "DF", "DF", "DF", "MF", "MF", "MF", "MF", "FW", "FW", "FW"]))
        self.assertFalse(is_legal_xi(["GK", "DF", "DF", "MF", "MF", "MF", "MF", "MF", "FW", "FW", "FW"]))

        xi = pd.DataFrame(
            [
                {"player": "XI GK", "position": "GK", "pts_if_plays": 4.0, "play_probability": 0.0},
                {"player": "DF1", "position": "DF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "DF2", "position": "DF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "DF3", "position": "DF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "MF1", "position": "MF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "MF2", "position": "MF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "MF3", "position": "MF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "MF4", "position": "MF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "FW1", "position": "FW", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "FW2", "position": "FW", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "FW3", "position": "FW", "pts_if_plays": 3.0, "play_probability": 1.0},
            ]
        )
        bench = pd.DataFrame(
            [
                {"player": "Bench DF", "position": "DF", "pts_if_plays": 5.0, "play_probability": 1.0},
                {"player": "Bench GK", "position": "GK", "pts_if_plays": 3.5, "play_probability": 1.0},
            ]
        )
        played = {name: name != "XI GK" for name in list(xi["player"]) + list(bench["player"])}
        final_xi, events = apply_autosub(xi, bench, played=played)
        self.assertEqual(events[0]["in"], "Bench GK")
        self.assertIn("Bench GK", final_xi["player"].tolist())
        self.assertNotIn("Bench DF", final_xi["player"].tolist())

    def test_autosub_skips_illegal_outfield_swap_and_uses_next_bench(self) -> None:
        from src.autosub import apply_autosub

        xi = pd.DataFrame(
            [
                {"player": "XI GK", "position": "GK", "pts_if_plays": 4.0},
                {"player": "DF1", "position": "DF", "pts_if_plays": 3.0},
                {"player": "DF2", "position": "DF", "pts_if_plays": 3.0},
                {"player": "DF3", "position": "DF", "pts_if_plays": 3.0},
                {"player": "MF1", "position": "MF", "pts_if_plays": 3.0},
                {"player": "MF2", "position": "MF", "pts_if_plays": 3.0},
                {"player": "MF3", "position": "MF", "pts_if_plays": 3.0},
                {"player": "MF4", "position": "MF", "pts_if_plays": 3.0},
                {"player": "MF5", "position": "MF", "pts_if_plays": 3.0},
                {"player": "FW1", "position": "FW", "pts_if_plays": 3.0},
                {"player": "FW2", "position": "FW", "pts_if_plays": 3.0},
            ]
        )
        bench = pd.DataFrame(
            [
                {"player": "Bench FW", "position": "FW", "pts_if_plays": 5.0},
                {"player": "Bench DF", "position": "DF", "pts_if_plays": 4.0},
            ]
        )
        played = {p: p != "DF1" for p in list(xi["player"]) + list(bench["player"])}
        final_xi, events = apply_autosub(xi, bench, played=played)
        self.assertEqual(events[0]["in"], "Bench DF")
        self.assertIn("Bench DF", final_xi["player"].tolist())

    def test_expected_squad_points_rewards_useful_bench_cover(self) -> None:
        from src.autosub import expected_squad_points

        xi = pd.DataFrame(
            [
                {"player": "Risky GK", "position": "GK", "pts_if_plays": 5.0, "play_probability": 0.2},
                {"player": "DF1", "position": "DF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "DF2", "position": "DF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "DF3", "position": "DF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "MF1", "position": "MF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "MF2", "position": "MF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "MF3", "position": "MF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "MF4", "position": "MF", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "FW1", "position": "FW", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "FW2", "position": "FW", "pts_if_plays": 3.0, "play_probability": 1.0},
                {"player": "FW3", "position": "FW", "pts_if_plays": 3.0, "play_probability": 1.0},
            ]
        )
        covered = expected_squad_points(
            xi,
            pd.DataFrame(
                [{"player": "Cover GK", "position": "GK", "pts_if_plays": 4.0, "play_probability": 1.0}]
            ),
            draws=200,
        )
        uncovered = expected_squad_points(
            xi,
            pd.DataFrame(
                [{"player": "Useless FW", "position": "FW", "pts_if_plays": 8.0, "play_probability": 1.0}]
            ),
            draws=200,
        )
        self.assertGreater(covered["expected_pts"], uncovered["expected_pts"])

    def test_merge_prices_preserves_established_super_lig_apps(self) -> None:
        from src.load_prices import merge_prices

        stats = pd.DataFrame(
            [
                {
                    "player": "Muhammed Şengezer",
                    "team": "Başakşehir",
                    "position": "GK",
                    "projected_pts": 4.0,
                    "form_apps": 1.0,
                    "base_apps": 1.0,
                    "current_apps": 1.0,
                    "prev_apps": 33.0,
                    "established_sl_apps": 33.0,
                    "data_src": "super_lig",
                    "reason": "CS",
                    "min_per_app": 90.0,
                    "cs_raw": 0.43,
                    "cs_after_fixture": 0.37,
                    "fixture_cs_mult": 0.85,
                }
            ]
        )
        prices = pd.DataFrame(
            [
                {
                    "player_name": "Muhammed Şengezer",
                    "display_name": "Muhammed Şengezer",
                    "team": "Başakşehir",
                    "position": "GK",
                    "price_m": 4.5,
                }
            ]
        )
        merged = merge_prices(stats, prices)
        self.assertEqual(float(merged.iloc[0]["prev_apps"]), 33.0)
        self.assertEqual(float(merged.iloc[0]["established_sl_apps"]), 33.0)
        self.assertAlmostEqual(float(merged.iloc[0]["cs_raw"]), 0.43)

    def test_established_super_lig_history_skips_external_prior(self) -> None:
        from src.fetch_external import apply_external_priors

        frame = pd.DataFrame(
            [
                {
                    "player": "Şengezer",
                    "team": "Başakşehir",
                    "position": "GK",
                    "price_m": 4.5,
                    "projected_pts": 4.2,
                    "stats_player": "Muhammed Şengezer",
                    "form_apps": 1.0,
                    "current_apps": 1.0,
                    "prev_apps": 33.0,
                    "established_sl_apps": 33.0,
                    "base_apps": 1.0,
                    "min_per_app": 90.0,
                    "data_src": "super_lig",
                },
                {
                    "player": "Yeni Kaleci",
                    "team": "X",
                    "position": "GK",
                    "price_m": 5.0,
                    "projected_pts": 0.0,
                    "stats_player": "",
                    "form_apps": 0.0,
                    "current_apps": 0.0,
                    "prev_apps": 0.0,
                    "established_sl_apps": 0.0,
                    "base_apps": 0.0,
                    "min_per_app": 0.0,
                    "data_src": "",
                },
            ]
        )
        out = apply_external_priors(frame, max_fetch=0)
        self.assertEqual(out.loc[0, "data_src"], "super_lig")
        self.assertEqual(float(out.loc[0, "prev_apps"]), 33.0)

    def test_goalkeeper_opener_keeps_external_saves_as_component_blend(self) -> None:
        from src.fetch_external import apply_external_priors

        frame = pd.DataFrame(
            [
                {
                    "player": "Alexander Nübel",
                    "team": "Beşiktaş",
                    "position": "GK",
                    "price_m": 5.0,
                    "projected_pts": 3.26,
                    "stats_player": "Alexander Nübel",
                    "form_apps": 1.0,
                    "current_apps": 1.0,
                    "current_minutes": 90.0,
                    "prev_apps": 0.0,
                    "established_sl_apps": 0.0,
                    "base_apps": 1.0,
                    "min_per_app": 90.0,
                    "share_60": 1.0,
                    "saves_pa": 1.0,
                    "current_saves_pa": 1.0,
                    "team_cs_base": 0.40,
                    "fixture_cs_mult": 1.0,
                    "fixture_attack_mult": 1.0,
                    "data_src": "super_lig",
                }
            ]
        )
        external = {
            "projected_pts": 6.0,
            "reason": "Bundesliga",
            "data_src": "external_prior",
            "ext_league": "Bundesliga",
            "ext_saves": 109,
            "ext_cs": 11,
            "ext_saves_pa": 3.30,
            "ext_cs_rate": 0.333,
            "translated_saves_pa": 3.0,
            "translated_cs_rate": 0.27,
            "share_60": 1.0,
            "min_per_app": 90.0,
            "league_calibration_level": "global",
            "league_calibration_note": "global küçültme",
            "rating": 7.0,
        }
        with (
            patch(
                "src.fetch_external.resolve_one",
                return_value={"player": {"id": 1}},
            ),
            patch(
                "src.fetch_external.project_external_player",
                return_value=external,
            ),
        ):
            out = apply_external_priors(frame, max_fetch=1)

        self.assertEqual(out.loc[0, "data_src"], "external_blend")
        self.assertEqual(float(out.loc[0, "ext_saves"]), 109)
        self.assertEqual(float(out.loc[0, "ext_cs"]), 11)
        # 1 maç: w_current = 1/5 = 0.2 → 0.2*1 + 0.8*3 = 2.6
        self.assertAlmostEqual(float(out.loc[0, "saves_pa"]), 2.6, places=3)
        self.assertLess(float(out.loc[0, "projected_pts"]), 6.0)
        self.assertGreater(float(out.loc[0, "projected_pts"]), 3.26)
        self.assertIn("kurtarış:", str(out.loc[0, "reason"]))

    def test_goalkeeper_season_transition_weights_are_monotonic(self) -> None:
        from src.scoring import blend_goalkeeper_components, season_sample_weight

        weights = [
            season_sample_weight(n, 4.0) for n in (0, 1, 3, 6, 10, 20)
        ]
        self.assertEqual(weights[0], 0.0)
        self.assertAlmostEqual(weights[1], 1 / 5)
        self.assertAlmostEqual(weights[2], 3 / 7)
        self.assertAlmostEqual(weights[3], 6 / 10)
        self.assertAlmostEqual(weights[4], 10 / 14)
        self.assertAlmostEqual(weights[5], 20 / 24)
        self.assertEqual(weights, sorted(weights))

        packs = [
            blend_goalkeeper_components(
                current_saves_pa=1.0,
                prior_saves_pa=3.3,
                current_sample=float(n),
                team_cs=0.40,
            )
            for n in (0, 1, 3, 6, 10, 20)
        ]
        saves = [p["saves_pa"] for p in packs]
        prior_w = [p["w_saves_prior"] for p in packs]
        self.assertEqual(saves, sorted(saves, reverse=True))
        self.assertEqual(prior_w, sorted(prior_w, reverse=True))
        self.assertAlmostEqual(packs[0]["saves_pa"], 3.3, places=3)
        self.assertAlmostEqual(packs[-1]["saves_pa"], 20 / 24 * 1.0 + 4 / 24 * 3.3, places=3)

    def test_nubel_and_sengezer_component_priors_compare_fairly(self) -> None:
        from src.scoring import blend_goalkeeper_components

        nubel = blend_goalkeeper_components(
            current_saves_pa=1.0,
            prior_saves_pa=109 / 33,
            current_sample=1.0,
            team_cs=0.40,
            personal_prior_cs=11 / 33,
            fixture_cs_mult=1.01,
        )
        sengezer = blend_goalkeeper_components(
            current_saves_pa=4.0,
            prior_saves_pa=105 / 33,
            current_sample=1.0,
            team_cs=0.43,
            personal_prior_cs=12 / 33,
            fixture_cs_mult=0.85,
        )
        self.assertGreater(nubel["saves_pa"], 2.5)
        self.assertGreater(sengezer["saves_pa"], 3.0)
        self.assertAlmostEqual(nubel["personal_prior_cs"], 11 / 33, places=4)
        self.assertAlmostEqual(sengezer["personal_prior_cs"], 12 / 33, places=4)
        # CS puanı kişisel Bundesliga/SL CS'den değil takım+fikstürden gelir.
        self.assertAlmostEqual(nubel["cs_raw"], 0.40, places=4)
        self.assertAlmostEqual(sengezer["cs_raw"], 0.43, places=4)
        self.assertLess(sengezer["cs_after_fixture"], sengezer["cs_raw"])

    def test_limited_super_lig_history_can_blend_external_prior(self) -> None:
        from src.fetch_external import apply_external_priors

        frame = pd.DataFrame(
            [
                {
                    "player": "Sınırlı Örnek",
                    "team": "X",
                    "position": "MF",
                    "price_m": 7.0,
                    "projected_pts": 4.0,
                    "stats_player": "Sınırlı Örnek",
                    "form_apps": 1.0,
                    "current_apps": 1.0,
                    "prev_apps": 5.0,
                    "established_sl_apps": 5.0,
                    "base_apps": 1.0,
                    "min_per_app": 75.0,
                    "data_src": "super_lig",
                }
            ]
        )
        external = {
            "projected_pts": 6.0,
            "reason": "dış lig",
            "data_src": "external_prior",
        }
        with (
            patch(
                "src.fetch_external.resolve_one",
                return_value={"player": {"id": 1}},
            ),
            patch(
                "src.fetch_external.project_external_player",
                return_value=external,
            ),
        ):
            out = apply_external_priors(frame, max_fetch=1)

        self.assertEqual(out.loc[0, "data_src"], "external_blend")
        self.assertAlmostEqual(float(out.loc[0, "projected_pts"]), 4.9)
        self.assertEqual(float(out.loc[0, "prev_apps"]), 5.0)

    def test_soft_early_form_keeps_strong_super_lig_base(self) -> None:
        from src.scoring import soft_early_form_rates

        form = _empty_rates()
        form.update({"apps": 1.0, "gls_pa": 1.0, "ast_pa": 0.0, "xg_pa": 0.9})
        base = _empty_rates()
        base.update({"apps": 30.0, "gls_pa": 0.25, "ast_pa": 0.20, "xg_pa": 0.22})
        softened = soft_early_form_rates(form, base, 1.0, 30.0)
        self.assertLess(softened["gls_pa"], 0.40)
        self.assertGreater(softened["gls_pa"], base["gls_pa"])

    def test_fixture_cs_floor_preserves_strong_team_edge(self) -> None:
        from src.config import FIXTURE_CS_FLOOR

        self.assertGreaterEqual(FIXTURE_CS_FLOOR, 0.85)
        basaksehir = 0.43 * 0.85
        rizespor = 0.30 * 1.0
        self.assertGreater(basaksehir, rizespor)

    def test_formation_rescore_prefers_three_forwards_when_bench_star_is_valuable(self) -> None:
        from src.optimize import rescore_formations

        rows = []
        teams = list("ABCDEFGHIJKLMNO")
        rows += [
            {"player": "GK1", "team": teams[0], "position": "GK", "price_m": 4.0, "pts_if_plays": 4.0, "play_probability": 0.95, "projected_pts": 3.8},
            {"player": "GK2", "team": teams[1], "position": "GK", "price_m": 4.0, "pts_if_plays": 2.0, "play_probability": 0.4, "projected_pts": 0.8},
        ]
        for i in range(5):
            rows.append(
                {
                    "player": f"DF{i}",
                    "team": teams[2 + i],
                    "position": "DF",
                    "price_m": 4.5,
                    "pts_if_plays": 3.2 - i * 0.05,
                    "play_probability": 0.9,
                    "projected_pts": 2.8,
                }
            )
        for i in range(5):
            rows.append(
                {
                    "player": f"MF{i}",
                    "team": teams[7 + i] if 7 + i < len(teams) else f"T{i}",
                    "position": "MF",
                    "price_m": 5.0,
                    "pts_if_plays": 3.5 - i * 0.1,
                    "play_probability": 0.9,
                    "projected_pts": 3.0,
                }
            )
        rows += [
            {"player": "Osimhen", "team": "Galatasaray", "position": "FW", "price_m": 14.0, "pts_if_plays": 7.0, "play_probability": 0.95, "projected_pts": 6.6},
            {"player": "Shomu", "team": "Fenerbahce", "position": "FW", "price_m": 10.0, "pts_if_plays": 6.0, "play_probability": 0.95, "projected_pts": 5.7},
            {"player": "Talisca", "team": "Fenerbahce", "position": "FW", "price_m": 8.5, "pts_if_plays": 5.8, "play_probability": 0.92, "projected_pts": 5.3},
        ]
        squad = pd.DataFrame(rows)
        formations = {
            "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
            "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
            "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
        }
        best = rescore_formations(squad, formations, autosub_draws=80)
        self.assertIn(best["formation"], ("3-4-3", "4-3-3"))
        self.assertIn("Talisca", best["xi"]["player"].tolist())
        compared = [
            row["formation"] for row in best["formation_comparisons"]
        ]
        self.assertEqual(len(compared), len(set(compared)))

    def test_card_budget_raises_threshold_when_rights_are_scarce(self) -> None:
        from src.manager_cards import choose_manager_card, opportunity_threshold

        self.assertGreater(
            opportunity_threshold(6.0, remaining=2, weeks_left=20),
            opportunity_threshold(6.0, remaining=10, weeks_left=34),
        )
        hold = choose_manager_card(
            [{"card": "Tripleks Kaptan", "extra_pts": 6.4, "why": "3x"}],
            remaining=2,
            weeks_left=20,
        )
        self.assertFalse(hold["use"])
        self.assertEqual(hold["remaining"], 2)

    def test_record_card_use_persists_history_and_decrements_total(self) -> None:
        from src.manager_cards import (
            load_card_state,
            record_card_use,
            set_cards_remaining,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "card_state.json"
            set_cards_remaining(
                8,
                path=path,
                season=2026,
                weeks_left=30,
            )
            state = record_card_use(
                "Tripleks Kaptan",
                path=path,
                season=2026,
                week=5,
                weeks_left=29,
            )
            loaded = load_card_state(path, season=2026)

        self.assertEqual(state["remaining"], 7)
        self.assertEqual(state["used"], 3)
        self.assertEqual(loaded["remaining"], 7)
        self.assertEqual(loaded["history"][-1]["card"], "Tripleks Kaptan")
        self.assertEqual(loaded["history"][-1]["week"], 5)

        with TemporaryDirectory() as directory:
            invalid_path = Path(directory) / "card_state.json"
            with self.assertRaises(ValueError):
                set_cards_remaining(11, path=invalid_path, season=2026)

    def test_card_state_cli_does_not_run_weekly_pipeline(self) -> None:
        from src.main import main

        state = {
            "remaining": 8,
            "budget": 10,
            "season": 2026,
            "weeks_left": 30,
        }
        with patch(
            "src.manager_cards.set_cards_remaining",
            return_value=state,
        ) as update:
            code = main(
                [
                    "--set-cards-remaining",
                    "8",
                    "--season",
                    "2026",
                    "--weeks-left",
                    "30",
                ]
            )

        self.assertEqual(code, 0)
        update.assert_called_once()

    def test_league_evaluation_promotes_only_better_than_identity(self) -> None:
        from calibrate_leagues import evaluate_leagues_forward

        samples = []
        for year in (2020, 2021, 2022, 2023, 2024):
            for i in range(20):
                src = 0.4 + 0.02 * i
                samples.append(
                    {
                        "player_id": year * 100 + i,
                        "source_tournament_id": 17,
                        "source_league": "Premier League",
                        "position": "FW",
                        "target_season_start": year,
                        "source_minutes": 2000.0,
                        "target_minutes": 1800.0,
                        "source_apps": 34.0,
                        "target_apps": 30.0,
                        "source_goals": src * 20,
                        "source_assists": 5.0,
                        "target_goals": src * 14,
                        "target_assists": 3.0,
                        "source_xg": src * 18,
                        "target_xg": src * 13,
                        "source_xa": 4.0,
                        "target_xa": 3.0,
                        "source_key_passes": 30.0,
                        "target_key_passes": 22.0,
                        "source_shots_on_target": 40.0,
                        "target_shots_on_target": 28.0,
                        "source_saves": 0.0,
                        "target_saves": 0.0,
                        "source_clean_sheets": 0.0,
                        "target_clean_sheets": 0.0,
                        "source_goals_conceded": 0.0,
                        "target_goals_conceded": 0.0,
                        "source_yellow_cards": 2.0,
                        "target_yellow_cards": 2.0,
                        "source_red_cards": 0.0,
                        "target_red_cards": 0.0,
                        "source_rating": 7.0,
                        "target_rating": 6.8,
                        "source_has_xg": True,
                        "target_has_xg": True,
                        "source_has_xa": True,
                        "target_has_xa": True,
                        "source_has_key_passes": True,
                        "target_has_key_passes": True,
                        "source_has_shots_on_target": True,
                        "target_has_shots_on_target": True,
                    }
                )
            for i in range(8):
                src = 0.5 + 0.03 * i
                samples.append(
                    {
                        "player_id": 9000 + year * 10 + i,
                        "source_tournament_id": 99,
                        "source_league": "Noise League",
                        "position": "FW",
                        "target_season_start": year,
                        "source_minutes": 1600.0,
                        "target_minutes": 1500.0,
                        "source_apps": 28.0,
                        "target_apps": 25.0,
                        "source_goals": src * 25,
                        "source_assists": 2.0,
                        "target_goals": 2.0 + (i % 3),
                        "target_assists": 1.0,
                        "source_xg": src * 20,
                        "target_xg": 2.0,
                        "source_xa": 2.0,
                        "target_xa": 1.0,
                        "source_key_passes": 20.0,
                        "target_key_passes": 10.0,
                        "source_shots_on_target": 30.0,
                        "target_shots_on_target": 12.0,
                        "source_saves": 0.0,
                        "target_saves": 0.0,
                        "source_clean_sheets": 0.0,
                        "target_clean_sheets": 0.0,
                        "source_goals_conceded": 0.0,
                        "target_goals_conceded": 0.0,
                        "source_yellow_cards": 1.0,
                        "target_yellow_cards": 1.0,
                        "source_red_cards": 0.0,
                        "target_red_cards": 0.0,
                        "source_rating": 7.2,
                        "target_rating": 6.5,
                        "source_has_xg": True,
                        "target_has_xg": True,
                        "source_has_xa": True,
                        "target_has_xa": True,
                        "source_has_key_passes": True,
                        "target_has_key_passes": True,
                        "source_has_shots_on_target": True,
                        "target_has_shots_on_target": True,
                    }
                )

        report = evaluate_leagues_forward(samples)
        self.assertIn("summary", report)
        self.assertIn("leagues", report)
        noise = report["leagues"].get("99", {})
        if noise:
            promotes = [
                row.get("promote")
                for pos in (noise.get("positions") or {}).values()
                for row in pos.values()
            ]
            self.assertTrue(any(p is False for p in promotes) or not promotes)


if __name__ == "__main__":
    unittest.main()
