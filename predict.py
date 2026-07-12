

import os
import sys
import warnings
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, classification_report

warnings.filterwarnings("ignore")

CACHE_DIR = "data_cache"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIXTURES_PATH = os.path.join(CACHE_DIR, "data.csv")

# normalizes the historical results.csv team names
NAME_MAP = 
{
    "USA": "United States", "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland", "Türkiye": "Turkey",
    "Cape Verde": "Cabo Verde", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Congo DR": "DR Congo", "Congo": "Republic of the Congo",
}

# maps data.csv team names -> the normalized results.csv names
FIXTURE_NAME_MAP =
{
    "IR Iran": "Iran", "Korea Republic": "South Korea", "Türkiye": "Turkey",
    "Congo DR": "DR Congo", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao", "USA": "United States",
    "Cape Verde": "Cabo Verde",
}

FEATURES = [
    "neutral", "tournament_weight", "home_elo", "away_elo", "elo_diff",
    "home_win5", "away_win5", "home_gd5", "away_gd5",
    "home_win10", "away_win10", "home_rest_days", "away_rest_days",
    "h2h_n", "h2h_home_winrate", "h2h_home_gd",
]

TRAIN_START = "2006-01-01"
VAL_START = "2023-01-01"
MATCH_WEIGHT = 4          # FIFA World Cup
MATCH_NEUTRAL = True      # 2026 group games at neutral US/CA/MX venues for these teams

ELO_BASE = 1500.0
ELO_K = 32
ELO_HOME_BONUS = 60

# palette (matches your existing reel charts)
INK, MUTE, GRID = "#1a1a2e", "#8a8a9e", "#e8e8ee"
ORANGE, BLUE, GRAY = "#ff6b18", "#1f6feb", "#9aa0a6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── data loading ────────────────────────────────────────────────────────────────
def fetch_results():
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "results.csv")
    if not os.path.exists(path):
        resp = requests.get(RESULTS_URL, timeout=120)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
    return pd.read_csv(path)


def normalize_country(name):
    return NAME_MAP.get(name, name) if isinstance(name, str) else name


def load_results():
    r = fetch_results()
    r["home_team"] = r["home_team"].map(normalize_country)
    r["away_team"] = r["away_team"].map(normalize_country)
    r["date"] = pd.to_datetime(r["date"])
    r = r.dropna(subset=["home_score", "away_score"]).copy()
    r["home_score"] = r["home_score"].astype(int)
    r["away_score"] = r["away_score"].astype(int)
    r["neutral"] = r["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    return r.sort_values("date").reset_index(drop=True)


# ── feature engineering ─────────────────────────────────────────────────────────
def tournament_weight(name):
    t = str(name).lower()
    if "fifa world cup" in t and "qualif" not in t:
        return 4
    if "qualif" in t:
        return 3
    big = ["uefa nations", "copa america", "afc asian cup", "africa cup",
           "concacaf", "uefa euro", "confederations"]
    if any(tok in t for tok in big):
        return 3
    if "friendly" in t:
        return 1
    return 2


def add_label_and_context(r):
    r = r.copy()
    r["label"] = np.where(r["home_score"] > r["away_score"], 0,
                          np.where(r["home_score"] == r["away_score"], 1, 2))
    r["tournament_weight"] = r["tournament"].map(tournament_weight)
    return r


def compute_elo(r):
    r = r.sort_values("date").reset_index(drop=True)
    rating, home_pre, away_pre = {}, np.zeros(len(r)), np.zeros(len(r))
    for i, row in r.iterrows():
        rh = rating.get(row.home_team, ELO_BASE)
        ra = rating.get(row.away_team, ELO_BASE)
        home_pre[i], away_pre[i] = rh, ra
        bonus = 0 if row.neutral == 1 else ELO_HOME_BONUS
        exp_home = 1 / (1 + 10 ** (-((rh + bonus) - ra) / 400))
        score_home = 1.0 if row.label == 0 else (0.5 if row.label == 1 else 0.0)
        margin = abs(int(row.home_score) - int(row.away_score))
        mult = np.log(max(margin, 1) + 1) * (2.2 / (abs(rh - ra) * 0.001 + 2.2))
        rating[row.home_team] = rh + ELO_K * mult * (score_home - exp_home)
        rating[row.away_team] = ra + ELO_K * mult * ((1 - score_home) - (1 - exp_home))
    r["home_elo"], r["away_elo"] = home_pre, away_pre
    r["elo_diff"] = home_pre - away_pre
    return r, rating


def per_team_long(r):
    home = pd.DataFrame({"date": r["date"].values, "team": r["home_team"].values,
                         "opp": r["away_team"].values, "gf": r["home_score"].values,
                         "ga": r["away_score"].values})
    away = pd.DataFrame({"date": r["date"].values, "team": r["away_team"].values,
                         "opp": r["home_team"].values, "gf": r["away_score"].values,
                         "ga": r["home_score"].values})
    long = pd.concat([home, away], ignore_index=True)
    long["result"] = np.where(long["gf"] > long["ga"], 1.0,
                              np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["gd"] = long["gf"] - long["ga"]
    return long


def add_form_features(r):
    long = per_team_long(r).sort_values(["team", "date"]).reset_index(drop=True)
    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["result_lag"] = long.groupby("team")["result"].shift(1)
    long["gd_lag"] = long.groupby("team")["gd"].shift(1)
    long["win5"] = long.groupby("team")["result_lag"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["gd5"] = long.groupby("team")["gd_lag"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    long["win10"] = long.groupby("team")["result_lag"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days
    form = long[["date", "team", "win5", "gd5", "win10", "rest_days"]].drop_duplicates(["date", "team"])
    r = r.merge(form.rename(columns={"team": "home_team", "win5": "home_win5", "gd5": "home_gd5",
                                     "win10": "home_win10", "rest_days": "home_rest_days"}),
                on=["date", "home_team"], how="left")
    r = r.merge(form.rename(columns={"team": "away_team", "win5": "away_win5", "gd5": "away_gd5",
                                     "win10": "away_win10", "rest_days": "away_rest_days"}),
                on=["date", "away_team"], how="left")
    return r


def add_h2h_features(r):
    long = per_team_long(r).sort_values(["team", "opp", "date"]).reset_index(drop=True)
    g = long.groupby(["team", "opp"])
    long["h2h_n"] = g.cumcount()
    long["h2h_winrate"] = g["result"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    long["h2h_gd"] = g["gd"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    h2h = long[["date", "team", "opp", "h2h_n", "h2h_winrate", "h2h_gd"]].drop_duplicates(["date", "team", "opp"])
    r = r.merge(h2h.rename(columns={"team": "home_team", "opp": "away_team",
                                    "h2h_winrate": "h2h_home_winrate", "h2h_gd": "h2h_home_gd"}),
                on=["date", "home_team", "away_team"], how="left")
    return r


def build_dataset(r):
    r = add_label_and_context(r)
    r, final_elo = compute_elo(r)
    r = add_form_features(r)
    r = add_h2h_features(r)
    return r, final_elo


# ── model ───────────────────────────────────────────────────────────────────────
def split_by_date(ds, train_start, val_start, cutoff):
    train = ds[(ds["date"] >= pd.Timestamp(train_start)) & (ds["date"] < pd.Timestamp(val_start))].copy()
    val = ds[(ds["date"] >= pd.Timestamp(val_start)) & (ds["date"] < pd.Timestamp(cutoff))].copy()
    return train, val


def train_model(train, val):
    X_train, y_train = train[FEATURES].astype(float), train["label"].astype(int)
    X_val, y_val = val[FEATURES].astype(float), val["label"].astype(int)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=600,
        learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, eval_metric="mlogloss", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, X_val, y_val


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)
    pred = proba.argmax(axis=1)
    base = np.tile(np.bincount(y_val, minlength=3) / len(y_val), (len(y_val), 1))
    print(f"  Validation accuracy : {accuracy_score(y_val, pred):.3f}")
    print(f"  Validation log-loss : {log_loss(y_val, proba):.3f}  (baseline {log_loss(y_val, base, labels=[0,1,2]):.3f})")


# ── prediction helpers ──────────────────────────────────────────────────────────
def form_as_of(long, team, asof):
    sub = long[(long["team"] == team) & (long["date"] < pd.Timestamp(asof))].sort_values("date")
    if len(sub) == 0:
        return {"win5": 0.5, "gd5": 0.0, "win10": 0.5, "rest_days": 30.0}
    l5, l10 = sub.tail(5), sub.tail(10)
    return {"win5": float(l5["result"].mean()), "gd5": float((l5["gf"] - l5["ga"]).mean()),
            "win10": float(l10["result"].mean()),
            "rest_days": float((pd.Timestamp(asof) - sub["date"].max()).days)}


def h2h_as_of(long, team, opp, asof):
    sub = long[(long["team"] == team) & (long["opp"] == opp) & (long["date"] < pd.Timestamp(asof))]
    if len(sub) == 0:
        return 0.0, np.nan, np.nan
    return float(len(sub)), float(sub["result"].mean()), float(sub["gd"].mean())


def build_match_row(long, final_elo, home, away, neutral, weight, asof):
    hf, af = form_as_of(long, home, asof), form_as_of(long, away, asof)
    he, ae = final_elo.get(home, ELO_BASE), final_elo.get(away, ELO_BASE)
    n, wr, gd = h2h_as_of(long, home, away, asof)
    row = {"neutral": int(neutral), "tournament_weight": weight, "home_elo": he, "away_elo": ae,
           "elo_diff": he - ae, "home_win5": hf["win5"], "away_win5": af["win5"],
           "home_gd5": hf["gd5"], "away_gd5": af["gd5"], "home_win10": hf["win10"],
           "away_win10": af["win10"], "home_rest_days": hf["rest_days"],
           "away_rest_days": af["rest_days"], "h2h_n": n, "h2h_home_winrate": wr, "h2h_home_gd": gd}
    return pd.DataFrame([row])[FEATURES].astype(float)


def predict_symmetric(model, long, final_elo, a, b, asof, neutral, weight):
    p_ab = model.predict_proba(build_match_row(long, final_elo, a, b, neutral, weight, asof))[0]
    p_ba = model.predict_proba(build_match_row(long, final_elo, b, a, neutral, weight, asof))[0]
    p_a = (p_ab[0] + p_ba[2]) / 2.0
    p_d = (p_ab[1] + p_ba[1]) / 2.0
    p_b = (p_ab[2] + p_ba[0]) / 2.0
    tot = p_a + p_d + p_b
    return p_a / tot, p_d / tot, p_b / tot


# ── fixtures ──────────────────────────────────────────────────────────────────
def map_fixture_name(name):
    name = name.strip()
    return FIXTURE_NAME_MAP.get(name, name)


def _side_matches(user_input, raw_name):
    """True if the user's typed team matches a fixture side (by raw or mapped name)."""
    u = user_input.strip().lower()
    return u in {raw_name.strip().lower(), map_fixture_name(raw_name).strip().lower()}


def find_fixture(team_a, team_b):
    """Find the single fixture for the two named teams (order doesn't matter)."""
    fx = pd.read_csv(FIXTURES_PATH)
    for _, row in fx.iterrows():
        if " v " not in str(row["teams"]):
            continue
        left, right = [p.strip() for p in str(row["teams"]).split(" v ")]
        forward = _side_matches(team_a, left) and _side_matches(team_b, right)
        reverse = _side_matches(team_a, right) and _side_matches(team_b, left)
        if forward or reverse:
            return {"match": row.get("match_number", ""), "group": row.get("group", ""),
                    "stadium": row.get("stadium", ""), "date": row.get("date_dt", ""),
                    "home_disp": left, "away_disp": right,
                    "home": map_fixture_name(left), "away": map_fixture_name(right)}
    return None


def list_team_names():
    fx = pd.read_csv(FIXTURES_PATH)
    names = set()
    for t in fx["teams"]:
        if " v " in str(t):
            for p in str(t).split(" v "):
                p = p.strip()
                if not any(w in p.lower() for w in ["winner", "runner", "third", "place", "group"]):
                    names.add(p)
    return sorted(names)


# ── chart ───────────────────────────────────────────────────────────────────────
def make_chart(m, p_home, p_draw, p_away, slate_date, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = [f"{m['home_disp']}\nwin", "Draw", f"{m['away_disp']}\nwin"]
    vals = [p_home, p_draw, p_away]
    colors = [ORANGE, GRAY, BLUE]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.2, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=16, fontweight="bold")
    ax.set_ylim(0, max(vals) * 100 + 12)
    ax.set_ylabel("Win probability (%)")
    sub = f"{slate_date}  ·  {m['group']}  ·  {m['stadium']}"
    ax.set_title(f"{m['home_disp']} vs {m['away_disp']}\n{sub}", fontsize=13)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    safe = f"{m['home_disp']}_vs_{m['away_disp']}".replace(" ", "_").replace("/", "-")
    path = os.path.join(out_dir, f"viz_{safe}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def tag_match(top_prob, p_home, p_away, home_elo, away_elo):
    favorite_is_home = p_home >= p_away
    fav_elo_is_home = home_elo >= away_elo
    upset = (favorite_is_home != fav_elo_is_home)
    if top_prob >= 0.60:
        strength = "LOCK"
    elif top_prob >= 0.45:
        strength = "LEAN"
    else:
        strength = "TOSS-UP"
    return strength + ("  ⚠️ UPSET PICK" if upset else "")


# ── main ────────────────────────────────────────────────────────────────────────
def get_teams_from_args():
    """Two team names from the command line, or ask for them interactively."""
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    print("Enter the two teams to predict (e.g. Saudi Arabia / Uruguay).")
    a = input("  Team 1: ").strip()
    b = input("  Team 2: ").strip()
    return a, b


def main():
    team_a, team_b = get_teams_from_args()

    print("\nLoading data + building features ...")
    results = load_results()
    dataset, final_elo = build_dataset(results)
    valid_teams = set(results["home_team"]) | set(results["away_team"])
    long = per_team_long(results)

    m = find_fixture(team_a, team_b)
    if m is None:
        print(f"\n  Couldn't find a World Cup match between '{team_a}' and '{team_b}'.")
        print("  Check spelling. Teams in the tournament:")
        print("   " + ", ".join(list_team_names()))
        return
    if m["home"] not in valid_teams or m["away"] not in valid_teams:
        print(f"\n  That match isn't predictable yet (a team is still a placeholder, e.g. a knockout slot).")
        return

    match_date = m["date"]
    print(f"Training model (data up to {match_date} ...")
    train, val = split_by_date(dataset, TRAIN_START, VAL_START, match_date)
    model, X_val, y_val = train_model(train, val)

    p_home, p_draw, p_away = predict_symmetric(
        model, long, final_elo, m["home"], m["away"], match_date, MATCH_NEUTRAL, MATCH_WEIGHT)
    outcomes = [(m["home_disp"], p_home), ("Draw", p_draw), (m["away_disp"], p_away)]
    pick, conf = max(outcomes, key=lambda x: x[1])
    he, ae = final_elo.get(m["home"], ELO_BASE), final_elo.get(m["away"], ELO_BASE)
    tag = tag_match(conf, p_home, p_away, he, ae)

    out_dir = os.path.join("predictions", str(match_date))
    os.makedirs(out_dir, exist_ok=True)
    chart = make_chart(m, p_home, p_draw, p_away, match_date, out_dir)

    # print the single result
    print("\n" + "=" * 60)
    print(f"  {m['home_disp']} vs {m['away_disp']}")
    print(f"  {match_date}  ·  {m['group']}  ·  {m['stadium']}")
    print("=" * 60)
    print(f"  {m['home_disp']:<22} win   {p_home*100:>5.1f}%")
    print(f"  {'Draw':<22}       {p_draw*100:>5.1f}%")
    print(f"  {m['away_disp']:<22} win   {p_away*100:>5.1f}%")
    print("-" * 60)
    print(f"  PICK: {pick}  ({conf*100:.1f}%)   [{tag}]")
    print("=" * 60)
    print(f"  Chart saved -> {chart}\n")


if __name__ == "__main__":
    main()
