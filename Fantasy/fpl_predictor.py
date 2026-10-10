#!/usr/bin/env python3
"""
AI Fantasy Premier League points predictor
===============================================================================
Data
  * FPL API (Opta-derived): xG, xA, xGI, xGC, ICT, BPS, minutes ... per player per match
  * Set-piece roles: penalty / direct free-kick / corner takers (penalties_order etc.)
  * Past seasons from the public vaastav archive (extra training data)
  * Bookmaker odds (football-data.co.uk, free): 1X2 + over/under 2.5 -> implied team goals,
    win probability and clean-sheet probability, for training AND for the next gameweek
  * Optional anytime-goalscorer odds (any provider) via a CSV -> blended into projections

    python fpl_predictor.py --horizon 5 --squad
    python fpl_predictor.py --scorer-odds scorers.csv      # player,team,odds
    python fpl_predictor.py --demo --scorer-odds demo       # offline test with simulated data
"""
import argparse
import difflib
import json
import re
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, least_squares, milp
from scipy.stats import poisson, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
API = "https://fantasy.premierleague.com/api"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}

# ============================================================================== data
NUMERIC = ["total_points", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "bonus", "bps",
           "influence", "creativity", "threat", "ict_index", "expected_goals", "expected_assists",
           "expected_goal_involvements", "expected_goals_conceded", "saves", "value", "selected"]
ROLL_COLS = ["total_points", "minutes", "started", "expected_goals", "expected_assists", "expected_goal_involvements",
             "expected_goals_conceded", "bps", "ict_index", "threat", "creativity", "influence", "bonus",
             "goals_scored", "assists", "clean_sheets", "saves"]
WINDOWS = (3, 6, 12)
SETPIECE_COLS = ["penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order"]


def _get(url, retries=4):
    import requests
    for i in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Could not fetch {url}")


def load_live(cache_dir="cache", refresh=False, max_age_h=6):
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True, parents=True)
    f = cache / "snapshot.json"
    if f.exists() and not refresh and time.time() - f.stat().st_mtime < max_age_h * 3600:
        print(f"Using cached data ({f})")
        return json.loads(f.read_text())
    print("Downloading FPL data (bootstrap, fixtures, per-player histories)...")
    boot, fixtures = _get(f"{API}/bootstrap-static/"), _get(f"{API}/fixtures/")
    ids = [e["id"] for e in boot["elements"] if e["minutes"] > 0]
    with ThreadPoolExecutor(8) as ex:
        summaries = list(ex.map(lambda i: _get(f"{API}/element-summary/{i}/"), ids))
    snap = {"bootstrap": boot, "fixtures": fixtures, "history": [h for s in summaries for h in s["history"]]}
    f.write_text(json.dumps(snap))
    return snap


def set_piece_flags(df):
    """Penalty / free-kick / corner roles as 0-1 flags (FPL: order 1 = first choice)."""
    for c in SETPIECE_COLS:
        if c not in df:
            df[c] = np.nan
    o = lambda c: pd.to_numeric(df[c], errors="coerce")
    df["pen1"] = (o("penalties_order") == 1).astype(int)
    df["pen2"] = (o("penalties_order") == 2).astype(int)
    df["dfk1"] = (o("direct_freekicks_order") == 1).astype(int)
    df["ck1"] = (o("corners_and_indirect_freekicks_order") == 1).astype(int)
    return df


def clean_history(h):
    for c in NUMERIC:
        h[c] = pd.to_numeric(h[c], errors="coerce") if c in h else 0.0
    h[NUMERIC] = h[NUMERIC].fillna(0.0)
    h["kickoff_time"] = pd.to_datetime(h["kickoff_time"], utc=True)
    h["started"] = (h["minutes"] >= 60).astype(float)
    h["sel_rank"] = h.groupby("round")["selected"].rank(pct=True)
    return h


def prepare(snap):
    b = snap["bootstrap"]
    players = pd.DataFrame(b["elements"])
    teams = pd.DataFrame(b["teams"]).set_index("id")
    fixtures = pd.DataFrame(snap["fixtures"])
    h = pd.DataFrame(snap["history"])
    keep = ["id", "team", "element_type"] + [c for c in SETPIECE_COLS if c in players]
    h = h.merge(players[keep].rename(columns={"id": "element"}), on="element")
    return players, teams, fixtures, set_piece_flags(clean_history(h))


ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
POS_MAP = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}


def load_archive(seasons, cache_dir="cache", with_odds=True):
    """Past seasons as extra training rows (+ their bookmaker odds). Each season keeps its own team table;
    player ids are namespaced per season. Set-piece roles are end-of-season snapshots (slightly optimistic)."""
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True, parents=True)
    frames, odds_parts = [], []
    for i, season in enumerate(seasons):
        try:
            data = {}
            for key, url in (("gw", f"{ARCHIVE}/{season}/gws/merged_gw.csv"), ("teams", f"{ARCHIVE}/{season}/teams.csv"),
                             ("raw", f"{ARCHIVE}/{season}/players_raw.csv")):
                p = cache / f"{season}_{key}.csv"
                if not p.exists():
                    pd.read_csv(url).to_csv(p, index=False)
                data[key] = pd.read_csv(p)
            t, m, raw = data["teams"], data["gw"], data["raw"]
            m = m[m["position"].isin(POS_MAP)].copy()
            m["team"] = m["team"].map(dict(zip(t["name"], t["id"])))
            m["element_type"] = m["position"].map(POS_MAP)
            sp = raw[["id"] + [c for c in SETPIECE_COLS if c in raw]].rename(columns={"id": "element"})
            m = m.merge(sp, on="element", how="left")
            m["element"] = m["element"] + (i + 1) * 100_000
            m["was_home"] = m["was_home"].astype(str).str.lower().eq("true")
            m = m.dropna(subset=["team", "opponent_team"])
            m[["team", "opponent_team"]] = m[["team", "opponent_team"]].astype(int)
            m = set_piece_flags(clean_history(m))
            m = add_matchup(rolling_features(m, shift=1), t.set_index("id"))
            frames.append(m.assign(season=season))
            print(f"  archive {season}: {len(m)} player-matches")
            if with_odds:
                try:
                    code = season[2:4] + season[5:7]
                    rawodds = fetch_csv(f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv", cache, None)
                    odds_parts.append(odds_table(rawodds, team_lookup(t["name"], t["id"]), season))
                except Exception as ex:
                    print(f"  odds for {season} unavailable: {ex}")
        except Exception as ex:
            print(f"  could not load archive season {season}: {ex}")
    return (pd.concat(frames, ignore_index=True) if frames else None), odds_parts


# ============================================================================== odds
ALIASES = {"manunited": "manutd", "manchesterunited": "manutd", "manchestercity": "mancity", "tottenham": "spurs",
           "tottenhamhotspur": "spurs", "wolverhampton": "wolves", "wolverhamptonwanderers": "wolves",
           "newcastleunited": "newcastle", "westhamunited": "westham", "brightonandhovealbion": "brighton",
           "brightonhovealbion": "brighton", "leedsunited": "leeds", "leicestercity": "leicester",
           "norwichcity": "norwich", "nottinghamforest": "nottmforest", "nottmforest": "nottmforest",
           "sheffieldunited": "sheffieldutd", "sheffutd": "sheffieldutd", "westbromwichalbion": "westbrom",
           "ipswichtown": "ipswich", "lutontown": "luton", "afcbournemouth": "bournemouth", "hull": "hullcity"}


def norm_team(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return ALIASES.get(s, s)


def team_lookup(names, ids):
    return {norm_team(n): int(i) for n, i in zip(names, ids)}


def fetch_csv(url, cache, max_age_h):
    """Download a CSV with disk cache (max_age_h=None -> cache forever). Falls back to stale cache."""
    p = Path(cache) / (re.sub(r"\W+", "_", url) + ".csv")
    if p.exists() and (max_age_h is None or time.time() - p.stat().st_mtime < max_age_h * 3600):
        return pd.read_csv(p)
    try:
        df = pd.read_csv(url, encoding_errors="ignore", on_bad_lines="skip")
        df.to_csv(p, index=False)
        return df
    except Exception:
        if p.exists():
            return pd.read_csv(p)
        raise


_G = np.arange(11)
_TOT = np.add.outer(_G, _G)


def _poisson_probs(lh, la):
    M = np.outer(poisson.pmf(_G, lh), poisson.pmf(_G, la))
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum(), M[_TOT >= 3].sum()


def solve_lambdas(p_h, p_d, p_a, p_over=None):
    """Find the Poisson goal rates (home, away) that reproduce the market's 1X2 and over-2.5 probabilities."""
    def resid(x):
        pH, pD, pA, pO = _poisson_probs(*x)
        r = [pH - p_h, pD - p_d, pA - p_a]
        r.append(pO - p_over if p_over is not None else 0.05 * (x[0] + x[1] - 2.75))
        return r
    return least_squares(resid, [1.5, 1.2], bounds=([0.05, 0.05], [6, 6])).x


def _first(raw, names):
    out = pd.Series(np.nan, index=raw.index)
    for n in names:
        if n in raw:
            out = out.fillna(pd.to_numeric(raw[n], errors="coerce"))
    return out


def odds_table(raw, name2id, season):
    """football-data.co.uk rows -> one row per (home, away) with implied goal rates and 1X2 probabilities."""
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    keys, bad = list(name2id), set()

    def to_id(n):
        k = norm_team(n)
        if k in name2id:
            return name2id[k]
        m = difflib.get_close_matches(k, keys, n=1, cutoff=0.8)
        if m:
            return name2id[m[0]]
        bad.add(n)
        return np.nan
    t = pd.DataFrame({"home": raw["HomeTeam"].map(to_id), "away": raw["AwayTeam"].map(to_id),
                      "oh": _first(raw, ["AvgH", "B365H", "PSH"]), "od": _first(raw, ["AvgD", "B365D", "PSD"]),
                      "oa": _first(raw, ["AvgA", "B365A", "PSA"]), "oo": _first(raw, ["Avg>2.5", "B365>2.5", "P>2.5"]),
                      "ou": _first(raw, ["Avg<2.5", "B365<2.5", "P<2.5"])}).dropna(subset=["home", "away", "oh", "od", "oa"])
    if bad:
        print(f"  odds {season}: unmatched team names {sorted(bad)} (add to ALIASES)")
    rows = []
    for r in t.itertuples():
        p = np.array([1 / r.oh, 1 / r.od, 1 / r.oa])
        p = p / p.sum()                                   # remove bookmaker margin proportionally
        po = None
        if not (np.isnan(r.oo) or np.isnan(r.ou)):
            po = (1 / r.oo) / (1 / r.oo + 1 / r.ou)
        lh, la = solve_lambdas(p[0], p[1], p[2], po)
        rows.append((int(r.home), int(r.away), lh, la, p[0], p[1], p[2]))
    out = pd.DataFrame(rows, columns=["home", "away", "lam_h", "lam_a", "p_h", "p_d", "p_a"])
    out = out.drop_duplicates(["home", "away"], keep="last")
    out["season"] = season
    print(f"  odds {season}: {len(out)} matches")
    return out


def load_current_odds(teams, fixtures, cache_dir="cache"):
    """Played matches of this season (E0.csv) + upcoming fixtures (fixtures.csv, usually next round only)."""
    cache = Path(cache_dir)
    ko = pd.to_datetime(fixtures["kickoff_time"], utc=True).min()
    y = ko.year if ko.month >= 7 else ko.year - 1
    code = f"{y % 100:02d}{(y + 1) % 100:02d}"
    raws = []
    for url in (f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv", "https://www.football-data.co.uk/fixtures.csv"):
        try:
            d = fetch_csv(url, cache, 6)
            raws.append(d[d["Div"] == "E0"] if "Div" in d else d)
        except Exception as ex:
            print(f"  could not download {url}: {ex}")
    if not raws:
        return None
    return odds_table(pd.concat(raws, ignore_index=True), team_lookup(teams["name"], teams.index), "current")


def attach_odds(df, tbl):
    """Add odds-implied features for the row's own team: expected goals for/against, win probability."""
    df = df.copy()
    home = df["was_home"].values == 1
    df["_h"] = np.where(home, df["team"], df["opponent_team"]).astype(int)
    df["_a"] = np.where(home, df["opponent_team"], df["team"]).astype(int)
    if tbl is not None and len(tbl):
        df = df.merge(tbl.rename(columns={"home": "_h", "away": "_a"}), on=["season", "_h", "_a"], how="left")
        home = df["was_home"].values == 1
        df["imp_gf"] = np.where(home, df["lam_h"], df["lam_a"])
        df["imp_ga"] = np.where(home, df["lam_a"], df["lam_h"])
        df["imp_win"] = np.where(home, df["p_h"], df["p_a"])
        df = df.drop(columns=["lam_h", "lam_a", "p_h", "p_d", "p_a"])
    else:
        for c in ("imp_gf", "imp_ga", "imp_win"):
            df[c] = np.nan
    return derive_odds_cols(df.drop(columns=["_h", "_a"]))


def derive_odds_cols(d):
    d["imp_total"] = d["imp_gf"] + d["imp_ga"]
    d["imp_cs"] = np.exp(-d["imp_ga"])            # P(opponent scores 0) under the implied Poisson rate
    return d


PROXY_X = ["own_att", "own_def", "opp_att", "opp_def", "was_home"]


def fit_odds_proxy(df):
    """Where odds are missing (later gameweeks, unmatched games) estimate them from team strengths."""
    ok = df["imp_gf"].notna()
    if ok.sum() < 200:
        return None
    return {t: Ridge(alpha=1.0).fit(df.loc[ok, PROXY_X], df.loc[ok, t]) for t in ("imp_gf", "imp_ga", "imp_win")}


def apply_odds_proxy(df, proxies):
    if proxies:
        for t, m in proxies.items():
            miss = df[t].isna()
            if miss.any():
                df.loc[miss, t] = m.predict(df.loc[miss, PROXY_X])
    return derive_odds_cols(df)


# ============================================================================== features
def add_matchup(df, teams):
    home = df["was_home"].astype(bool).values
    m = lambda col, key: df[key].map(teams[col])
    df["own_att"] = np.where(home, m("strength_attack_home", "team"), m("strength_attack_away", "team"))
    df["own_def"] = np.where(home, m("strength_defence_home", "team"), m("strength_defence_away", "team"))
    df["opp_att"] = np.where(home, m("strength_attack_away", "opponent_team"), m("strength_attack_home", "opponent_team"))
    df["opp_def"] = np.where(home, m("strength_defence_away", "opponent_team"), m("strength_defence_home", "opponent_team"))
    df["att_edge"], df["def_edge"] = df["own_att"] - df["opp_def"], df["own_def"] - df["opp_att"]
    df["was_home"] = home.astype(int)
    for k, name in POS.items():
        df[f"is_{name}"] = (df["element_type"] == k).astype(int)
    return df


def rolling_features(df, shift):
    """shift=1: features use only PRIOR matches (training, no leakage). shift=0: includes latest match."""
    df = df.sort_values(["element", "kickoff_time"]).copy()
    g = df.groupby("element")
    base = (g[ROLL_COLS].shift(1) if shift else df[ROLL_COLS]).assign(element=df["element"])
    bg = base.groupby("element")[ROLL_COLS]
    out = {}
    for w in WINDOWS:
        r = bg.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        out.update({f"{c}_r{w}": r[c] for c in ROLL_COLS})
    e = bg.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    out.update({f"{c}_s": e[c] for c in ROLL_COLS})
    out["n_prior"] = g.cumcount() + (0 if shift else 1)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


FORM = [f"{c}_r{w}" for w in WINDOWS for c in ROLL_COLS] + [f"{c}_s" for c in ROLL_COLS]
PEN_FEATS = ["pen1", "pen2", "dfk1", "ck1"]
ODDS_FEATS = ["imp_gf", "imp_ga", "imp_win", "imp_total", "imp_cs"]
BASE_FEATS = FORM + ["n_prior", "was_home", "own_att", "own_def", "opp_att", "opp_def", "att_edge", "def_edge",
                     "value", "sel_rank"] + [f"is_{n}" for n in POS.values()]
FEATURES = BASE_FEATS + PEN_FEATS + ODDS_FEATS


# ============================================================================== models
def make_model(seed=0, loss="squared_error"):
    return HistGradientBoostingRegressor(loss=loss, max_iter=350, learning_rate=0.04, max_depth=5,
                                         min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                                         random_state=seed)


def backtest(train_df, k):
    cur = train_df[train_df["season"] == "current"]
    rounds = sorted(cur["round"].unique())
    k = min(k, max(1, len(rounds) // 3))
    test_rounds = rounds[-k:]
    te = cur[cur["round"].isin(test_rounds)]
    tr = train_df[(train_df["season"] != "current") | (train_df["round"] < test_rounds[0])]
    if len(tr) < 300 or te.empty:
        print("Not enough history for a meaningful backtest yet - skipping.")
        return
    full = make_model().fit(tr[FEATURES], tr["total_points"])
    lite = make_model().fit(tr[BASE_FEATS], tr["total_points"])
    te = te.assign(p_full=full.predict(te[FEATURES]), p_lite=lite.predict(te[BASE_FEATS]),
                   p_base=te["total_points_r6"].fillna(0))
    rmse = lambda a, b: mean_squared_error(a, b) ** 0.5

    def block(d, title):
        print(f"{title:44s}{'MAE':>7s}{'RMSE':>7s}{'Spearman':>10s}")
        for name, c in (("Rolling avg (baseline)", "p_base"), ("AI model, no odds/penalty features", "p_lite"),
                        ("AI model, full (odds + penalties)", "p_full")):
            print(f"  {name:42s}{mean_absolute_error(d.total_points, d[c]):7.3f}{rmse(d.total_points, d[c]):7.3f}"
                  f"{spearmanr(d.total_points, d[c])[0]:10.3f}")
    print(f"\nBacktest on GW{test_rounds[0]}-{test_rounds[-1]} (trained on earlier rounds + past seasons)")
    block(te, f"All players (n={len(te)})")
    reg = te[te["minutes_r6"] >= 45]
    block(reg, f"Regular starters (n={len(reg)})")
    if te["imp_gf"].notna().mean() > 0:
        print(f"  (odds-implied features present for {te['imp_gf'].notna().mean():.0%} of test rows before proxy fill)")


def availability(row, first_event):
    status, chance = row["status"], row["chance_of_playing_next_round"]
    if status in ("u", "s", "n"):
        return 0.0
    p = 1.0 if (pd.isna(chance) and status == "a") else (0.5 if pd.isna(chance) else float(chance) / 100)
    return p if row["event"] == first_event else 0.5 + 0.5 * p


# ============================================================================== scorer odds (optional)
def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def match_scorer_odds(sodds, players, teams):
    """Map rows of (player, [team], decimal odds) to FPL element ids. Returns Series element -> raw implied prob."""
    by_team = {}
    for r in players.itertuples():
        names = {norm_name(r.web_name)}
        fn, sn = getattr(r, "first_name", None), getattr(r, "second_name", None)
        if isinstance(sn, str):
            names.add(norm_name(sn))
            if isinstance(fn, str):
                names.add(norm_name(f"{fn} {sn}"))
        for n in names:
            by_team.setdefault(r.team, {}).setdefault(n, set()).add(r.id)
    tl = team_lookup(teams["name"], teams.index)
    got, miss = {}, []
    for r in sodds.itertuples():
        pn = norm_name(r.player)
        tname = getattr(r, "team", None)
        tid = tl.get(norm_team(tname)) if isinstance(tname, str) else None
        pools = [by_team.get(tid, {})] if tid else list(by_team.values())
        ids = set()
        for pool in pools:
            ids |= pool.get(pn, set())
        if not ids and pn:
            last = pn.split()[-1]
            for pool in pools:
                for k, v in pool.items():
                    if k.split()[-1] == last:
                        ids |= v
        if not ids and pn:
            allk = {k: v for pool in pools for k, v in pool.items()}
            c = difflib.get_close_matches(pn, list(allk), n=1, cutoff=0.85)
            if c:
                ids = allk[c[0]]
        if len(ids) == 1 and r.odds > 1:
            got[next(iter(ids))] = 1 / float(r.odds)
        else:
            miss.append(r.player)
    print(f"Scorer odds: matched {len(got)} players, skipped {len(miss)}"
          + (f" (e.g. {', '.join(map(str, miss[:5]))})" if miss else ""))
    return pd.Series(got, dtype=float)


def apply_scorer_odds(rows, praw, weight, first):
    """Shift a player's projection toward the bookmaker's view of whether he scores.
    The bookmaker margin is calibrated away so only the RELATIVE view across players is used; level comes
    from the model + team odds."""
    rows["p_raw"] = rows["element"].map(praw)
    first_rows = rows[rows["event"] == first].sort_values("kickoff_time").groupby("element").head(1).index
    mask = rows.index.isin(first_rows) & rows["p_raw"].notna()
    if mask.sum() == 0:
        return rows
    p_model = 1 - np.exp(-rows.loc[mask, "lam_goal"])
    margin = float(np.clip(rows.loc[mask, "p_raw"].mean() / p_model.mean(), 1.0, 1.8))
    p_book = (rows.loc[mask, "p_raw"] / margin).clip(upper=0.95)
    d_lam = (-np.log(1 - p_book) - rows.loc[mask, "lam_goal"]).clip(-0.5, 0.5)
    pts = rows.loc[mask, "element_type"].map(GOAL_PTS) + 1.0      # +1 ~ extra bonus a goal brings
    rows.loc[mask, "odds_adj"] = weight * d_lam * pts * rows.loc[mask, "p_avail"]
    rows.loc[mask, "xpts"] += rows.loc[mask, "odds_adj"]
    rows.loc[mask, "lam_goal"] += weight * d_lam
    print(f"Scorer odds applied to {mask.sum()} player-fixtures (calibrated margin x{margin:.2f}, weight {weight})")
    return rows


# ============================================================================== prediction
def predict_future(model, goal_model, h, players, teams, fixtures, horizon, odds_tbl, proxies, praw, weight):
    state = rolling_features(h, shift=0).groupby("element").tail(1)[["element"] + FORM + ["n_prior"]]
    keep = ["id", "web_name", "team", "element_type", "now_cost", "sel_rank", "status", "chance_of_playing_next_round"] \
        + [c for c in SETPIECE_COLS if c in players]
    pl = players[players["id"].isin(state["element"])].copy()
    pl["sel_rank"] = pd.to_numeric(pl["selected_by_percent"], errors="coerce").rank(pct=True)
    pl = set_piece_flags(pl)
    keep += PEN_FEATS

    fx = fixtures[(fixtures["finished"] == False) & fixtures["event"].notna()].copy()  # noqa: E712
    first = int(fx["event"].min())
    fx = fx[fx["event"] < first + horizon]
    sides = []
    for s, o in (("h", "a"), ("a", "h")):
        sides.append(pd.DataFrame({"event": fx["event"].astype(int).values,
                                   "kickoff_time": pd.to_datetime(fx["kickoff_time"], utc=True).values,
                                   "team": fx[f"team_{s}"].values, "opponent_team": fx[f"team_{o}"].values,
                                   "was_home": s == "h"}))
    rows = pd.concat(sides).merge(pl[keep], on="team").rename(columns={"id": "element", "now_cost": "value"})
    rows = add_matchup(rows.merge(state, on="element"), teams).reset_index(drop=True)
    rows["season"] = "current"
    rows = apply_odds_proxy(attach_odds(rows, odds_tbl), proxies)
    rows["xpts_raw"] = model.predict(rows[FEATURES])
    rows["lam_goal"] = np.clip(goal_model.predict(rows[FEATURES]), 0.001, None)
    rows["p_avail"] = rows.apply(lambda r: availability(r, first), axis=1)
    rows["xpts"] = rows["xpts_raw"] * rows["p_avail"]
    rows["odds_adj"] = 0.0
    if praw is not None and len(praw):
        rows = apply_scorer_odds(rows, praw, weight, first)
    rows["goals"] = rows["lam_goal"] * rows["p_avail"]
    rows["goals_first"] = np.where(rows["event"] == first, rows["goals"], 0.0)

    gw = rows.pivot_table(index="element", columns="event", values="xpts", aggfunc="sum").fillna(0)
    gw.columns = [f"GW{c}" for c in gw.columns]
    agg = rows.groupby("element").agg(exp_goals=("goals", "sum"), g1=("goals_first", "sum"))
    out = gw.join(agg).join(pl.set_index("id")[["web_name", "team", "element_type", "now_cost", "status", "pen1", "pen2"]])
    out["team"] = out["team"].map(teams["name"])
    out["pos"] = out["element_type"].map(POS)
    out["price"] = out["now_cost"] / 10
    out["pen"] = np.where(out["pen1"] == 1, "P1", np.where(out["pen2"] == 1, "P2", ""))
    out["p_score"] = 1 - np.exp(-out["g1"])
    out["xpts_total"] = gw.sum(axis=1)
    out["first_gw"] = out[f"GW{first}"]
    cols = ["web_name", "team", "pos", "price", "status", "pen"] + list(gw.columns) + ["xpts_total", "exp_goals"]
    out = out.sort_values("xpts_total", ascending=False)
    out.index.name = "id"
    return out, cols, first


def print_penalty_takers(players, teams):
    if "penalties_order" not in players:
        return
    p = players.copy()
    p["o"] = pd.to_numeric(p["penalties_order"], errors="coerce")
    p = p[p["o"].between(1, 2)]
    if p.empty:
        return
    print("\n=== Penalty takers (FPL set-piece data: P1 first choice, P2 backup) ===")
    for tid, g in sorted(p.groupby("team"), key=lambda x: teams.loc[x[0], "name"]):
        g = g.sort_values("o")
        print(f"  {teams.loc[tid, 'name']:16s} " + ", ".join(f"{r.web_name} (P{int(r.o)})" for r in g.itertuples()))


# ============================================================================== squad optimiser
def pick_squad(pred, first, budget=100.0):
    p = pred[pred["status"] != "u"].reset_index()
    n, pts = len(p), p["xpts_total"].values
    pos, team, cost = p["pos"].values, p["team"].values, p["price"].values
    A, lo, hi = [], [], []

    def add(vx, vy, l, u):
        A.append(np.concatenate([vx, vy])); lo.append(l); hi.append(u)
    z = np.zeros(n)
    add(np.ones(n), z, 15, 15)
    add(cost, z, 0, budget)
    for ps, cnt in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        add((pos == ps).astype(float), z, cnt, cnt)
    for t in set(team):
        add((team == t).astype(float), z, 0, 3)
    add(z, np.ones(n), 11, 11)
    for ps, l, u in (("GK", 1, 1), ("DEF", 3, 5), ("MID", 2, 5), ("FWD", 1, 3)):
        add(z, (pos == ps).astype(float), l, u)
    for i in range(n):
        v = np.zeros(n); v[i] = 1
        add(-v, v, -np.inf, 0)
    res = milp(-np.concatenate([0.1 * pts, pts]), constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.ones(2 * n), bounds=Bounds(0, 1))
    if not res.success:
        print("Squad optimiser failed:", res.message)
        return
    x, y = np.round(res.x[:n]).astype(bool), np.round(res.x[n:]).astype(bool)
    xi, bench = p[y], p[x & ~y]
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    show = lambda d: d.assign(o=d["pos"].map(order)).sort_values(["o", "xpts_total"], ascending=[True, False])
    print(f"\n=== Optimal squad (cost £{p[x]['price'].sum():.1f}m, XI xPts over horizon {xi['xpts_total'].sum():.1f}) ===")
    for _, r in show(xi).iterrows():
        print(f"  {r['pos']:3s} {r['web_name']:18s} {r['team']:16s} £{r['price']:4.1f}  {r['xpts_total']:6.1f} {r['pen']}")
    print("  --- bench ---")
    for _, r in show(bench).iterrows():
        print(f"  {r['pos']:3s} {r['web_name']:18s} {r['team']:16s} £{r['price']:4.1f}  {r['xpts_total']:6.1f} {r['pen']}")
    cap = xi.sort_values("first_gw", ascending=False).iloc[0]
    print(f"  Captain for GW{first}: {cap['web_name']} ({cap['first_gw']:.2f} xPts, x2)")


# ============================================================================== main
def main():
    ap = argparse.ArgumentParser(description="FPL points predictor")
    ap.add_argument("--horizon", type=int, default=5, help="gameweeks ahead to predict")
    ap.add_argument("--backtest-rounds", type=int, default=4)
    ap.add_argument("--refresh", action="store_true", help="ignore cached API data")
    ap.add_argument("--squad", action="store_true", help="also build the optimal 15-man squad")
    ap.add_argument("--seasons", nargs="*", default=["2024-25", "2025-26"],
                    help="past seasons for extra training data (no values = disable)")
    ap.add_argument("--no-odds", action="store_true", help="skip bookmaker odds features")
    ap.add_argument("--scorer-odds", default=None,
                    help="CSV with columns player,[team],odds (decimal anytime-goalscorer odds), or 'demo'")
    ap.add_argument("--odds-weight", type=float, default=0.6, help="0-1: how much to trust scorer odds vs model")
    ap.add_argument("--demo", action="store_true", help="use simulated data (offline)")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()

    odds_parts, sodds = [], None
    if a.demo:
        from demo_data import make_snapshot
        snap = make_snapshot()
    else:
        snap = load_live(refresh=a.refresh)
    players, teams, fixtures, h = prepare(snap)
    print(f"{h['element'].nunique()} players, {len(h)} player-matches, GW1-{int(h['round'].max())} played")

    data = add_matchup(rolling_features(h, shift=1), teams).assign(season="current")
    if a.demo:
        tl = team_lookup(teams["name"], teams.index)
        if not a.no_odds:
            odds_parts.append(odds_table(snap["odds_raw"], tl, "current"))
        if a.seasons:
            extra = []
            for i, sd in enumerate((11, 12)):
                s2 = make_snapshot(n_played=38, seed=sd)
                _, pt, _, ph = prepare(s2)
                ph["element"] += (i + 1) * 100_000
                extra.append(add_matchup(rolling_features(ph, shift=1), pt).assign(season=f"past{i}"))
                if not a.no_odds:
                    odds_parts.append(odds_table(s2["odds_raw"], tl, f"past{i}"))
            data = pd.concat(extra + [data], ignore_index=True)
        if a.scorer_odds == "demo":
            sodds = snap["scorer_odds"]
    else:
        if a.seasons:
            print("Loading past seasons for extra training data...")
            arch, parts = load_archive(a.seasons, with_odds=not a.no_odds)
            if arch is not None:
                data = pd.concat([arch, data], ignore_index=True)
                odds_parts += parts
        if not a.no_odds:
            print("Loading bookmaker odds for this season...")
            cur = load_current_odds(teams, fixtures)
            if cur is not None:
                odds_parts.append(cur)
    if a.scorer_odds and a.scorer_odds != "demo":
        sodds = pd.read_csv(a.scorer_odds)

    odds_tbl = pd.concat(odds_parts, ignore_index=True) if odds_parts else None
    data = attach_odds(data, odds_tbl)
    coverage = data["imp_gf"].notna().mean()
    proxies = fit_odds_proxy(data)
    data = apply_odds_proxy(data, proxies)
    print(f"Odds-implied features available for {coverage:.0%} of training rows"
          + ("" if odds_tbl is not None else " (odds disabled/unavailable)"))

    train = data[data["n_prior"] >= 3]
    backtest(train, a.backtest_rounds)
    model = make_model().fit(train[FEATURES], train["total_points"])
    goal_model = make_model(loss="poisson").fit(train[FEATURES], train["goals_scored"])

    praw = match_scorer_odds(sodds, players, teams) if sodds is not None else None
    pred, cols, first = predict_future(model, goal_model, h, players, teams, fixtures, a.horizon,
                                       odds_tbl, proxies, praw, a.odds_weight)
    pred[cols].round(2).to_csv(a.out)
    print(f"\nSaved {len(pred)} player projections -> {a.out}")

    print_penalty_takers(players, teams)
    pd.set_option("display.width", 220)
    print(f"\n=== Top 15 overall, GW{first}-GW{first + a.horizon - 1} ===")
    print(pred[cols].head(15).round(2).to_string())
    print(f"\n=== Most likely scorers in GW{first} (anytime-goal probability) ===")
    top = pred.sort_values("p_score", ascending=False).head(10)
    print(top.assign(p_score=(top["p_score"] * 100).round(1))[["web_name", "team", "pos", "pen", "p_score"]]
          .rename(columns={"p_score": "P(score)%"}).to_string())
    for ps in POS.values():
        print(f"\n--- Top 5 {ps} ---")
        print(pred[pred["pos"] == ps][cols].head(5).round(2).to_string())
    if a.squad:
        pick_squad(pred, first)


if __name__ == "__main__":
    main()
