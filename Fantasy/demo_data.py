"""Synthetic snapshot in the same schema as the official FPL API (+ football-data style odds).
Used by `fpl_predictor.py --demo` so the whole pipeline can be tested offline."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy.stats import poisson

POS_LAYOUT = [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
ATT_RATE = {1: 0.0, 2: 0.05, 3: 0.14, 4: 0.32}      # open-play xG per 90
AST_RATE = {1: 0.01, 2: 0.07, 3: 0.14, 4: 0.12}     # xA per 90
GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
PEN_RATE, PEN_XG = 0.13, 0.78                        # penalties awarded per match (scaled), Opta-style pen xG


def _odds_row(lh, la, rng, margin=1.05):
    lh, la = lh * np.exp(rng.normal(0, 0.08)), la * np.exp(rng.normal(0, 0.08))
    g = np.arange(11)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    p = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
    p = p / p.sum()
    po = M[np.add.outer(g, g) >= 3].sum()
    o = 1 / (p * margin)
    return {"AvgH": o[0], "AvgD": o[1], "AvgA": o[2], "Avg>2.5": 1 / (po * 1.04), "Avg<2.5": 1 / ((1 - po) * 1.04)}


def make_snapshot(n_played=10, seed=7):
    rng = np.random.default_rng(seed)
    n_teams = 20
    teams = []
    for t in range(1, n_teams + 1):
        a, d = int(rng.integers(1100, 1360)), int(rng.integers(1100, 1360))
        teams.append(dict(id=t, name=f"Team {t}", strength_attack_home=a + 40, strength_attack_away=a - 40,
                          strength_defence_home=d + 40, strength_defence_away=d - 40))
    tdf = {t["id"]: t for t in teams}

    elements, pid = [], 1
    for t in range(1, n_teams + 1):
        for k, pos in enumerate(POS_LAYOUT):
            skill = float(np.clip(rng.lognormal(0, 0.35), 0.4, 2.2))
            sp = (0.95 if k == 0 else 0.02) if pos == 1 else float(rng.choice([0.92, 0.65, 0.15], p=[0.45, 0.3, 0.25]))
            price = int(np.clip(38 + 38 * (skill - 0.4) + 6 * (pos - 1) + rng.normal(0, 4), 40, 130))
            elements.append(dict(id=pid, web_name=f"Player{pid}", team=t, element_type=pos, now_cost=price,
                                 selected_by_percent=str(round(float(np.clip(skill * 8 * rng.random(), 0.1, 60)), 1)),
                                 chance_of_playing_next_round=None, status="a", news="", minutes=1,
                                 penalties_order=None, direct_freekicks_order=None,
                                 corners_and_indirect_freekicks_order=None, _skill=skill, _sp=sp))
            pid += 1
    for e in rng.choice(elements, 12, replace=False):
        e["status"], e["chance_of_playing_next_round"] = ("d", 50) if rng.random() < 0.6 else ("u", 0)

    # set-piece takers
    for t in range(1, n_teams + 1):
        outfield = [e for e in elements if e["team"] == t and e["element_type"] > 1]
        outfield.sort(key=lambda e: -(ATT_RATE[e["element_type"]] * e["_skill"] * rng.uniform(0.85, 1.15)))
        outfield[0]["penalties_order"], outfield[1]["penalties_order"] = 1, 2
        mids = [e for e in outfield if e["element_type"] == 3]
        max(mids, key=lambda e: e["_skill"])["corners_and_indirect_freekicks_order"] = 1

    def pre_lambda(e, f):
        sp_eff = e["_sp"] * 0.9 + (1 - e["_sp"]) * 0.05
        lam = ATT_RATE[e["element_type"]] * e["_skill"] * f * sp_eff
        if e["penalties_order"] == 1:
            lam += PEN_RATE * f * PEN_XG * sp_eff
        elif e["penalties_order"] == 2:
            lam += PEN_RATE * f * PEN_XG * 0.1 * sp_eff
        return lam

    base = datetime(2026, 8, 15, 14, tzinfo=timezone.utc)
    fixtures, history, odds_rows, scorer_rows, fid = [], [], [], [], 1
    by_team = {t: [e for e in elements if e["team"] == t] for t in range(1, n_teams + 1)}

    for gw in range(1, 39):
        perm = rng.permutation(n_teams) + 1
        for i in range(0, n_teams, 2):
            h, a = int(perm[i]), int(perm[i + 1])
            ko = (base + timedelta(days=7 * (gw - 1))).isoformat().replace("+00:00", "Z")
            fx = dict(id=fid, event=gw, team_h=h, team_a=a, kickoff_time=ko, finished=gw <= n_played)
            fixtures.append(fx)
            fac = {}
            for team, opp, home in ((h, a, True), (a, h, False)):
                att = tdf[team]["strength_attack_home" if home else "strength_attack_away"]
                dfn = tdf[opp]["strength_defence_away" if home else "strength_defence_home"]
                fac[team] = (att / dfn) ** 1.5

            if gw <= n_played + 1:  # bookmaker odds exist for played matches and the next gameweek
                lam = {tm: sum(pre_lambda(e, fac[tm]) for e in by_team[tm]) for tm in (h, a)}
                odds_rows.append({"HomeTeam": f"Team {h}", "AwayTeam": f"Team {a}", **_odds_row(lam[h], lam[a], rng)})
            if gw == n_played + 1:
                for tm in (h, a):
                    for e in by_team[tm]:
                        p = 1 - np.exp(-pre_lambda(e, fac[tm]))
                        if p > 0.10:
                            scorer_rows.append({"player": e["web_name"], "team": f"Team {tm}",
                                                "odds": round(max(1.05, 1 / min(0.95, p * 1.12)), 2)})

            if gw <= n_played:
                sides = {}
                for team, opp, home in ((h, a, True), (a, h, False)):
                    f, rows = fac[team], []
                    for e in by_team[team]:
                        pos = e["element_type"]
                        if rng.random() < e["_sp"]:
                            mins = 90 if rng.random() < 0.75 else int(rng.integers(55, 89))
                        else:
                            mins = 20 if rng.random() < 0.25 else 0
                        xg = ATT_RATE[pos] * e["_skill"] * f * mins / 90
                        xa = AST_RATE[pos] * e["_skill"] * f * mins / 90
                        if e["corners_and_indirect_freekicks_order"] == 1:
                            xa *= 1.3
                        rows.append(dict(e=e, mins=mins, xg=xg, xa=xa, g=int(rng.poisson(xg)), a=int(rng.poisson(xa))))
                    if rng.random() < PEN_RATE * f:          # a penalty is awarded
                        for order in (1, 2):
                            tk = next((r for r in rows if r["e"]["penalties_order"] == order and r["mins"] > 0), None)
                            if tk:
                                tk["xg"] += PEN_XG
                                tk["g"] += int(rng.random() < PEN_XG)
                                break
                    sides[team] = rows
                score = {tm: sum(r["g"] for r in rows) for tm, rows in sides.items()}
                xgteam = {tm: sum(r["xg"] for r in rows) for tm, rows in sides.items()}
                fx["team_h_score"], fx["team_a_score"] = score[h], score[a]
                for team, opp, home in ((h, a, True), (a, h, False)):
                    conc = score[opp]
                    for r in sides[team]:
                        e, pos, mins = r["e"], r["e"]["element_type"], r["mins"]
                        bonus = int(min(3, r["g"] * 1.5 + r["a"] + rng.random() * 1.2)) if mins >= 60 else 0
                        cs = int(conc == 0 and mins >= 60)
                        saves = int(rng.poisson(2.5)) if pos == 1 and mins > 0 else 0
                        pts = (1 if mins > 0 else 0) + (1 if mins >= 60 else 0) + r["g"] * GOAL_PTS[pos] + r["a"] * 3 \
                            + cs * CS_PTS[pos] + bonus + saves // 3 - ((conc // 2) if pos <= 2 and mins >= 60 else 0)
                        history.append(dict(
                            element=e["id"], fixture=fid, opponent_team=opp, total_points=int(pts), was_home=home,
                            kickoff_time=ko, team_h_score=score[h], team_a_score=score[a], round=gw, minutes=mins,
                            goals_scored=r["g"], assists=r["a"], clean_sheets=cs, goals_conceded=conc, bonus=bonus,
                            bps=int(mins / 3 + r["g"] * 12 + r["a"] * 9 + cs * 8 + rng.integers(0, 8)),
                            influence=str(round(float(rng.random() * 40 * (mins > 0)), 1)),
                            creativity=str(round(float(r["xa"] * 120 + rng.random() * 8), 1)),
                            threat=str(round(float(r["xg"] * 250 + rng.random() * 8), 1)),
                            ict_index=str(round(float((r["xg"] + r["xa"]) * 40 + rng.random() * 3), 1)),
                            expected_goals=str(round(r["xg"], 2)), expected_assists=str(round(r["xa"], 2)),
                            expected_goal_involvements=str(round(r["xg"] + r["xa"], 2)),
                            expected_goals_conceded=str(round(xgteam[opp], 2)) if mins > 0 else "0.00",
                            saves=saves, value=e["now_cost"], selected=int(float(e["selected_by_percent"]) * 1e5)))
            fid += 1

    for e in elements:
        e.pop("_skill"); e.pop("_sp")
    return {"bootstrap": {"elements": elements, "teams": teams}, "fixtures": fixtures, "history": history,
            "odds_raw": pd.DataFrame(odds_rows), "scorer_odds": pd.DataFrame(scorer_rows)}
