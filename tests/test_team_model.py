import unittest

from src.optimize import _selection_value, _this_week_value
from src.scoring import (
    apply_context_adjustments,
    expected_points_from_rates,
    weighted_horizon_points,
)
from src.team_model import (
    blend_goal_expectation,
    fit_poisson_ratings,
    group_events_by_matchweek,
    predict_lambdas,
    ratings_from_standings,
)


def _attacker_rates() -> dict[str, float]:
    return {
        "apps": 12.0,
        "share_60": 0.9,
        "min_per_app": 82.0,
        "gls_pa": 0.45,
        "ast_pa": 0.20,
        "xg_pa": 0.42,
        "xa_pa": 0.18,
        "sot_pa": 1.4,
        "key_passes_pa": 1.2,
        "cs_rate": 0.0,
        "yc_pa": 0.1,
        "rc_pa": 0.0,
        "saves_pa": 0.0,
        "ga_pa": 0.0,
        "rating": 7.0,
        "pen_save_pa": 0.0,
        "pen_miss_pa": 0.0,
        "og_pa": 0.0,
        "bcc_pa": 0.0,
    }


class TeamModelTests(unittest.TestCase):
    def test_poisson_gives_higher_lambda_to_strong_attack_vs_weak_defence(self) -> None:
        matches = []
        for i in range(8):
            matches.append(
                {
                    "home": "gs",
                    "away": "weak",
                    "hg": 3.0,
                    "ag": 0.0,
                    "weight": 1.0,
                }
            )
            matches.append(
                {
                    "home": "weak",
                    "away": "gs",
                    "hg": 0.0,
                    "ag": 2.0,
                    "weight": 1.0,
                }
            )
            matches.append(
                {
                    "home": "mid",
                    "away": "other",
                    "hg": 1.0,
                    "ag": 1.0,
                    "weight": 1.0,
                }
            )
            matches.append(
                {
                    "home": "other",
                    "away": "mid",
                    "hg": 1.0,
                    "ag": 1.0,
                    "weight": 1.0,
                }
            )
        ratings = fit_poisson_ratings(matches)
        self.assertEqual(ratings["source"], "poisson")
        easy = predict_lambdas(ratings, "gs", "weak", home=True)
        hard = predict_lambdas(ratings, "weak", "gs", home=True)
        self.assertGreater(easy["lambda_for"], hard["lambda_for"])
        self.assertGreater(easy["p_cs"], hard["p_cs"])
        self.assertGreater(easy["attack_mult"], 1.0)
        self.assertLess(hard["attack_mult"], easy["attack_mult"])

    def test_home_advantage_raises_home_lambda(self) -> None:
        matches = []
        for _ in range(10):
            matches.append(
                {"home": "a", "away": "b", "hg": 2.0, "ag": 1.0, "weight": 1.0}
            )
            matches.append(
                {"home": "b", "away": "a", "hg": 2.0, "ag": 1.0, "weight": 1.0}
            )
        ratings = fit_poisson_ratings(matches)
        home = predict_lambdas(ratings, "a", "b", home=True)
        away = predict_lambdas(ratings, "a", "b", home=False)
        self.assertGreater(home["lambda_for"], away["lambda_for"])
        self.assertGreater(home["p_cs"], away["p_cs"])

    def test_standings_fallback_separates_attack_and_defence(self) -> None:
        ratings = ratings_from_standings(
            {
                "city": {"gf": 2.4, "ga": 0.7},
                "weak": {"gf": 0.8, "ga": 2.1},
            }
        )
        easy = predict_lambdas(ratings, "city", "weak", home=True)
        hard = predict_lambdas(ratings, "weak", "city", home=True)
        self.assertGreater(easy["lambda_for"], hard["lambda_for"])
        self.assertGreater(easy["p_cs"], hard["p_cs"])

    def test_player_share_tracks_team_lambda(self) -> None:
        hist = 0.40
        easy = blend_goal_expectation(
            hist, attack_mult=1.2, lambda_for=2.0, team_goal_rate=1.4
        )
        hard = blend_goal_expectation(
            hist, attack_mult=0.8, lambda_for=0.7, team_goal_rate=1.4
        )
        self.assertGreater(easy, hard)
        self.assertGreater(easy, hist * 0.8)

    def test_easy_fixture_scores_more_fantasy_points(self) -> None:
        rates = _attacker_rates()
        easy = expected_points_from_rates(
            rates,
            "FW",
            appearance=1.0,
            attack_mult=1.25,
            lambda_for=2.1,
            lambda_against=0.7,
            p_cs=0.50,
            team_goal_rate=1.4,
        )
        hard = expected_points_from_rates(
            rates,
            "FW",
            appearance=1.0,
            attack_mult=0.78,
            lambda_for=0.7,
            lambda_against=2.0,
            p_cs=0.14,
            team_goal_rate=1.4,
        )
        self.assertGreater(easy, hard)

    def test_horizon_weights_this_week_most(self) -> None:
        blended = weighted_horizon_points([10.0, 4.0, 1.0])
        self.assertGreater(blended, 7.0)
        self.assertLess(blended, 10.0)

    def test_selection_uses_horizon_captain_uses_this_week(self) -> None:
        import pandas as pd

        row = pd.Series(
            {
                "pts_if_plays": 8.0,
                "play_probability": 1.0,
                "projected_pts": 8.0,
                "selection_pts": 6.2,
            }
        )
        self.assertAlmostEqual(_this_week_value(row), 8.0)
        self.assertAlmostEqual(_selection_value(row), 6.2)

    def test_context_adjustment_builds_selection_pts(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "player": "Forvet",
                    "team": "A",
                    "position": "FW",
                    "projected_pts": 6.0,
                    "pts_week0": 6.0,
                    "pts_week1": 3.0,
                    "pts_week2": 3.0,
                    "availability": "AVAILABLE",
                    "form_apps": 6.0,
                    "min_per_app": 80.0,
                }
            ]
        )
        out = apply_context_adjustments(frame)
        self.assertIn("selection_pts", out.columns)
        self.assertLess(float(out.loc[0, "selection_pts"]), float(out.loc[0, "projected_pts"]))
        self.assertGreater(float(out.loc[0, "horizon_pts"]), 4.5)

    def test_fotmob_unplayed_rows_become_sofa_events(self) -> None:
        from src.fetch_fotmob import fotmob_rows_to_sofa_events

        rows = [
            {
                "home": {"name": "Galatasaray"},
                "away": {"name": "Kayserispor"},
                "status": {"finished": False, "utcTime": "2026-08-22T18:00:00.000Z"},
            },
            {
                "home": {"name": "Fenerbahçe"},
                "away": {"name": "Trabzonspor"},
                "status": {"finished": True, "utcTime": "2026-08-15T18:00:00.000Z"},
            },
        ]
        events = fotmob_rows_to_sofa_events(rows)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["homeTeam"]["name"], "Galatasaray")
        self.assertGreater(int(events[0]["startTimestamp"]), 0)

    def test_matchweek_grouping_splits_when_team_repeats(self) -> None:
        events = [
            {
                "startTimestamp": 1,
                "homeTeam": {"name": "A"},
                "awayTeam": {"name": "B"},
            },
            {
                "startTimestamp": 2,
                "homeTeam": {"name": "C"},
                "awayTeam": {"name": "D"},
            },
            {
                "startTimestamp": 3,
                "homeTeam": {"name": "A"},
                "awayTeam": {"name": "C"},
            },
        ]
        weeks = group_events_by_matchweek(events, weeks=3)
        self.assertEqual(len(weeks), 2)
        self.assertEqual(len(weeks[0]), 2)
        self.assertEqual(len(weeks[1]), 1)


if __name__ == "__main__":
    unittest.main()
