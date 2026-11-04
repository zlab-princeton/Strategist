"""Final pick of a search run against its reference strategy (paper Algorithm 1).

The reference R is the strategy named by --reference, measured in iteration 1
like any candidate. Every other measured (candidate, anchor, prompts) C is
compared with R question by question on the same splits:

    d   = accuracy(C) - accuracy(R) on the same questions, se = its standard error
    rho = cost(C) / cost(R) on the same questions,        se = delta-method error

A comparison is a win under the 2% rule (within 2 points / 2% is a tie):
    accuracy win:  d > 2pp and rho <= 1.02
    cost win:      rho < 0.98 and d >= -2pp

The pick (PICK_RULE = "hidden_best"): every strategy that wins on the validation
(blind) questions and again on the hidden questions, which the strategist never
sees, qualifies; the one with the highest hidden d is picked, ties by lower hidden
rho. If none qualifies the run reports R itself. Comparisons need at least MIN_N
shared questions per split. `choose_top3` (the top 3 by validation, confirmed on
hidden) and the "combined" route of `choose_alt` are kept as alternatives."""
from __future__ import annotations

import json
import math
from pathlib import Path

from strategist.prompt_slots import overrides_key

TIE_PP = 0.02
TIE_COST = 0.02
Z_CONFIRM = 2.13
CONFIRM_RULE = "two_percent"     # or "margin": a statistical margin instead of the 2% rule
TOP_K = 3
MIN_N = 20
PICK_RULE = "hidden_best"        # or "top3" or "combined" (see choose_alt)
SPLITS = ("blind", "hidden")


def gen_dir_name(candidate: str, anchor: str, prompts: dict | None = None) -> str:
    """Folder of one measured strategy under generations/iter_XX/. A blueprint run
    with other prompts is another strategy and gets its own folder (until
    2026-10-09 it shared the plain one, so a prompt-tweaked copy of the reference
    measured in the same iteration mixed both runs' rows into one samples.jsonl)."""
    name = f"{candidate}__{str(anchor).replace('/', '_')}"
    if prompts:
        name += "__p" + overrides_key(prompts)[:8]
    return name


def _samples(path: Path) -> dict[int, tuple[float, float]] | None:
    """idx -> (score, weighted_cost) of one split, or None when missing/ambiguous."""
    if not path.exists():
        return None
    out: dict[int, tuple[float, float]] = {}
    for line in path.open():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        i = int(x["idx"])
        if i in out:          # two strategies wrote into one directory: cannot tell them apart
            return None
        out[i] = (float(x.get("score") or 0.0), float(x.get("weighted_cost") or 0.0))
    return out


def measurements(run_dir: Path) -> list[dict]:
    """Every (candidate, anchor, prompts) the run measured itself (no cache hits),
    with per-question results per split."""
    out = []
    log = run_dir / "log.jsonl"
    if not log.exists():
        return out
    for line in log.open():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        it = e.get("iteration")
        if not it or e.get("final_test"):
            continue
        for c in e.get("candidates") or []:
            if any((c.get(f"{s}_summary") or {}).get("cached_from_iteration") for s in ("feedback",) + SPLITS):
                continue
            gen = run_dir / "generations" / f"iter_{int(it):02d}"
            d = gen / gen_dir_name(c["candidate"], c["anchor"], c.get("prompts"))
            if not d.exists():               # runs from before the per-prompts folders
                d = gen / gen_dir_name(c["candidate"], c["anchor"])
            q = {s: _samples(d / s / "samples.jsonl") for s in ("feedback",) + SPLITS}
            if any(q[s] is None for s in SPLITS):
                continue
            if sum(v[1] for s in SPLITS for v in q[s].values()) <= 0:
                continue                     # no LLM call: not a real measurement
            out.append({"iteration": int(it), "candidate": c["candidate"], "anchor": c["anchor"],
                        "prompts": c.get("prompts") or {}, "q": q})
    return out


def compare(cq: dict[int, tuple[float, float]], rq: dict[int, tuple[float, float]]) -> dict | None:
    keys = sorted(set(cq) & set(rq))
    n = len(keys)
    if n < 2:
        return None
    diff = [cq[k][0] - rq[k][0] for k in keys]
    d = sum(diff) / n
    se_d = math.sqrt(sum((x - d) ** 2 for x in diff) / (n - 1) / n)
    cc = [cq[k][1] for k in keys]
    rc = [rq[k][1] for k in keys]
    sr = sum(rc)
    rho = sum(cc) / max(sr, 1e-9)
    resid = [c - rho * r for c, r in zip(cc, rc)]
    se_rho = math.sqrt(sum(e * e for e in resid) * n / (n - 1)) / max(sr, 1e-9)
    fixed = sum(1 for k in keys if cq[k][0] > rq[k][0])
    broke = sum(1 for k in keys if cq[k][0] < rq[k][0])
    return {"n": n, "d": d, "se_d": se_d, "rho": rho, "se_rho": se_rho, "fixed": fixed, "broke": broke}


def win(s: dict, z: float, confirm: bool = False) -> str | None:
    """Accuracy win: d > 2pp, d - z*se > 0, rho <= 1.02. Cost win: rho + z*se < 0.98
    and accuracy held: d >= -2pp when ranking, d >= 0 when confirming. (A lower
    bound d - z*se >= -2pp cannot be met on 100 questions, where se(d) is 2-4pp:
    it would demand a measured gain of ~5pp from a pure cost win.)"""
    d, rho = round(s["d"], 9), round(s["rho"], 9)   # -2.0 points is a tie, not -2.0000000001
    if d > TIE_PP and d - z * s["se_d"] > 0 and rho <= 1 + TIE_COST:
        return "accuracy"
    if rho + z * s["se_rho"] < 1 - TIE_COST and d >= (0.0 if confirm else -TIE_PP):
        return "cost"
    return None


def _key(m: dict) -> tuple:
    return (m["candidate"], m["anchor"], json.dumps(m.get("prompts") or {}, sort_keys=True))


def choose(run_dir: Path, ref_blueprint: str | None, ref_anchor: str | None,
           exact_blueprints: list[str] | None = None) -> dict | None:
    """The final pick of a run (PICK_RULE) and how it was made, or None when the run
    has no usable reference."""
    if PICK_RULE == "top3":
        return choose_top3(run_dir, ref_blueprint, ref_anchor, exact_blueprints)
    return choose_alt(run_dir, ref_blueprint, ref_anchor, PICK_RULE, exact_blueprints)


def choose_top3(run_dir: Path, ref_blueprint: str | None, ref_anchor: str | None,
                exact_blueprints: list[str] | None = None) -> dict | None:
    """The top 3 by validation, confirmed on hidden. Also locates the reference
    for choose_alt."""
    ms = measurements(Path(run_dir))
    plain = [m for m in ms if not m["prompts"]]
    ref = next((m for m in plain if (m["candidate"], m["anchor"]) == (ref_blueprint, ref_anchor)), None)
    how_ref = "the reference strategy (--reference), measured in this run"
    if ref is None and exact_blueprints:
        # fallback: the most accurate exact baseline the run measured (blind + hidden)
        def acc(m):
            vals = [v[0] for s in SPLITS for v in m["q"][s].values()]
            cost = sum(v[1] for s in SPLITS for v in m["q"][s].values())
            return (sum(vals) / max(len(vals), 1), -cost)
        pool = [m for m in plain if m["candidate"] in exact_blueprints]
        if pool:
            ref = max(pool, key=acc)
            how_ref = "fallback: the most accurate of the given blueprints this run measured"
    if ref is None:
        return None
    ref_info = {k: ref[k] for k in ("iteration", "candidate", "anchor", "prompts")}
    for s in SPLITS:
        vals = list(ref["q"][s].values())
        ref_info[f"{s}_n"] = len(vals)
        ref_info[f"{s}_accuracy"] = sum(v[0] for v in vals) / max(len(vals), 1)
        ref_info[f"{s}_cost"] = sum(v[1] for v in vals)
    ref_info["how"] = how_ref

    seen = set()
    rows = []
    for m in ms:
        if not m["prompts"] and (m["candidate"], m["anchor"]) == (ref["candidate"], ref["anchor"]):
            continue                          # the reference or a re-measurement of it
        if _key(m) in seen:
            continue
        seen.add(_key(m))
        sb = compare(m["q"]["blind"], ref["q"]["blind"])
        sh = compare(m["q"]["hidden"], ref["q"]["hidden"])
        if not sb or not sh or sb["n"] < MIN_N or sh["n"] < MIN_N:
            continue
        rows.append({**{k: m[k] for k in ("iteration", "candidate", "anchor", "prompts")},
                     "blind": sb, "hidden": sh, "blind_route": win(sb, 0.0)})
    acc_rows = sorted((r for r in rows if r["blind_route"] == "accuracy"),
                      key=lambda r: -(r["blind"]["d"] - r["blind"]["se_d"]))
    cost_rows = sorted((r for r in rows if r["blind_route"] == "cost"),
                       key=lambda r: r["blind"]["rho"] + r["blind"]["se_rho"])
    shortlist = (acc_rows + cost_rows)[:TOP_K]
    pick, route = None, None
    for r in shortlist:
        r["hidden_route"] = (win(r["hidden"], 0.0) if CONFIRM_RULE == "two_percent"
                             else win(r["hidden"], Z_CONFIRM, confirm=True))
        if r["hidden_route"] and pick is None:
            pick, route = r, r["hidden_route"]
    slim = lambda r: {k: r[k] for k in ("iteration", "candidate", "anchor", "prompts", "blind", "hidden")  # noqa: E731
                      if k in r} | {k: r.get(k) for k in ("blind_route", "hidden_route")}
    result = {
        "rule": "paired_vs_reference",
        "reference": ref_info,
        "n_compared": len(rows),
        "shortlist": [slim(r) for r in shortlist],
        "route": f"{route} win over the reference, confirmed on hidden" if pick else "keep the reference",
    }
    chosen = pick or {**ref_info}
    result["pick"] = {k: chosen.get(k) for k in ("iteration", "candidate", "anchor", "prompts")}
    result["pick_is_reference"] = pick is None
    return result


ALT_ROUTES = ("hidden_best", "combined")


def choose_alt(run_dir: Path, ref_blueprint: str | None, ref_anchor: str | None, route: str,
               exact_blueprints: list[str] | None = None) -> dict | None:
    """Two final picks besides the top-3 shortlist.

    hidden_best: every measured strategy that wins under the 2% rule against the
        reference on blind and also on hidden (not only the top 3 by blind); the pick
        is the one with the highest hidden d, ties by lower hidden rho.
    combined:    training + blind + hidden questions pooled; every strategy that wins
        under the 2% rule on the pooled comparison; the pick is the one with the
        highest pooled d, ties by lower pooled rho.
    No such strategy: the reference."""
    assert route in ALT_ROUTES, route
    base = choose_top3(run_dir, ref_blueprint, ref_anchor, exact_blueprints)
    if base is None:
        return None
    ref_info = base["reference"]
    ms = measurements(Path(run_dir))
    is_ref = lambda m: not m["prompts"] and (m["candidate"], m["anchor"]) == (ref_info["candidate"], ref_info["anchor"])  # noqa: E731
    ref = next(m for m in ms if is_ref(m))
    pooled = lambda m: {(s, i): v for s in ("feedback",) + SPLITS for i, v in (m["q"].get(s) or {}).items()}  # noqa: E731
    seen, rows = set(), []
    for m in ms:
        if is_ref(m) or _key(m) in seen:
            continue
        seen.add(_key(m))
        sb = compare(m["q"]["blind"], ref["q"]["blind"])
        sh = compare(m["q"]["hidden"], ref["q"]["hidden"])
        if not sb or not sh or sb["n"] < MIN_N or sh["n"] < MIN_N:
            continue
        row = {**{k: m[k] for k in ("iteration", "candidate", "anchor", "prompts")}, "blind": sb, "hidden": sh,
               "blind_route": win(sb, 0.0), "hidden_route": win(sh, 0.0)}
        if route == "hidden_best":
            row["route"] = win(sb, 0.0) and win(sh, 0.0)
            row["rank_on"] = sh
        else:
            if m["q"].get("feedback") is None or ref["q"].get("feedback") is None:
                continue
            sc = compare(pooled(m), pooled(ref))
            row["combined"] = sc
            row["route"] = win(sc, 0.0)
            row["rank_on"] = sc
        rows.append(row)
    passing = sorted((r for r in rows if r["route"]),
                     key=lambda r: (-round(r["rank_on"]["d"], 9), r["rank_on"]["rho"]))
    pick = passing[0] if passing else None
    slim = lambda r: {k: r[k] for k in ("iteration", "candidate", "anchor", "prompts", "blind", "hidden", "combined",  # noqa: E731
                                        "route", "blind_route", "hidden_route") if k in r}
    chosen = pick or ref_info
    return {
        "rule": route,
        "reference": ref_info,
        "n_compared": len(rows),
        "n_passing": len(passing),
        "passing": [slim(r) for r in passing],
        "shortlist": [slim(r) for r in passing[:TOP_K]],   # the best 3, for the run's log lines
        "route": f"{pick['route']} win over the reference ({route})" if pick else "keep the reference",
        "pick": {k: chosen.get(k) for k in ("iteration", "candidate", "anchor", "prompts")},
        "pick_is_reference": pick is None,
    }


BLIND_VIEW_ENV = "STRATEGIST_BLIND_FEEDBACK"


def blind_direction_only() -> bool:
    """True when the strategist sees blind results only as a direction vs the
    reference (--blind-feedback direction). The
    pick above still ranks on the blind numbers; only the prompt is coarser."""
    import os
    return os.environ.get(BLIND_VIEW_ENV, "numbers") == "direction"


def direction(cq: dict[int, tuple[float, float]], rq: dict[int, tuple[float, float]]) -> str | None:
    """A candidate vs the reference on the same questions, as a direction only:
    within 2 points of accuracy or 2% of cost is "about the same". No MIN_N here:
    on the small benchmarks (3-18 blind rows) the direction is still what is shown."""
    s = compare(cq, rq)
    if not s:
        return None
    d, rho = round(s["d"], 9), round(s["rho"], 9)
    acc = "higher" if d > TIE_PP else "lower" if d < -TIE_PP else "about the same"
    cost = "lower" if rho < 1 - TIE_COST else "higher" if rho > 1 + TIE_COST else "about the same"
    return f"accuracy {acc}, cost {cost}"


def vs_reference_line(ref_feedback: dict[int, tuple[float, float]] | None, samples: list[dict]) -> str:
    """One line for the strategist: a candidate vs the reference on the feedback rows."""
    if not ref_feedback or not samples:
        return ""
    # eval_candidate keeps the samples in split order; idx on disk is that position
    cq = {i: (float(x.get("score") or 0.0), float(x.get("weighted_cost") or 0.0)) for i, x in enumerate(samples)}
    s = compare(cq, ref_feedback)
    if not s:
        return ""
    return (f"- vs the reference baseline on the same {s['n']} feedback rows: fixed {s['fixed']}, "
            f"broke {s['broke']} (net {s['d'] * 100:+.1f}pp ± {s['se_d'] * 100:.1f}), "
            f"cost {s['rho']:.2f}× (± {s['se_rho']:.2f})\n")


# ----------------------------------------------------------------- the bar to beat

def _remeasured(log_path: Path | None, blueprint: str | None, anchor: str) -> dict | None:
    """The run's own measurement of the reference: its plain `reuse` at its anchor,
    which the harness adds to iteration 1."""
    if log_path is None or not blueprint or not log_path.exists():
        return None
    for line in log_path.open():
        try:
            j = json.loads(line)
        except Exception:
            continue
        for c in j.get("candidates") or []:
            if c.get("candidate") != blueprint or c.get("anchor") != anchor or c.get("prompts"):
                continue
            fs, bs = c.get("feedback_summary") or {}, c.get("blind_summary") or {}
            if not fs.get("total"):
                continue
            n = max(int(fs.get("total", 0) or 0), 1)
            return {"iteration": j.get("iteration"), "fb_acc": float(fs.get("accuracy", 0) or 0),
                    "bl_acc": float(bs.get("accuracy", 0) or 0),
                    "cost": float(fs.get("total_weighted_cost", 0) or 0) / n}
    return None


def _vs_reference_blind(log_path: Path | None, blueprint: str, anchor: str, k: int = 3) -> list[str]:
    """The run's candidates vs the reference on the validation (blind) rows, best k."""
    if log_path is None or not log_path.exists():
        return []
    ms = measurements(log_path.parent)
    ref = next((m for m in ms if not m["prompts"] and (m["candidate"], m["anchor"]) == (blueprint, anchor)), None)
    if ref is None:
        return []
    rows = []
    for m in ms:
        if not m["prompts"] and (m["candidate"], m["anchor"]) == (blueprint, anchor):
            continue
        s = compare(m["q"]["blind"], ref["q"]["blind"])
        if s:
            rows.append((m, s, win(s, 0.0)))
    order = {"accuracy": 0, "cost": 1, None: 2}
    rows.sort(key=lambda r: (order[r[2]], -(r[1]["d"] - r[1]["se_d"]) if r[2] != "cost" else r[1]["rho"] + r[1]["se_rho"]))
    out = []
    for m, s, w in rows[:k]:
        tag = {"accuracy": "accuracy WIN", "cost": "cost WIN", None: "no win"}[w]
        pr = " +prompts" if m["prompts"] else ""
        out.append(f"  - `{m['candidate']}`{pr} @ `{m['anchor']}` (iter {m['iteration']}): blind {s['d'] * 100:+.1f}pp "
                   f"± {s['se_d'] * 100:.1f} (fixed {s['fixed']}, broke {s['broke']} of {s['n']}), cost "
                   f"{s['rho']:.2f}× ± {s['se_rho']:.2f} → {tag} on blind")
    return out


def render_reference_bar(ref_cell: tuple[str, str], run_log: Path | None) -> str:
    """The "bar to beat" section of the strategist prompt for a run with --reference."""
    bp, anchor = ref_cell
    lines = [
        f"- **Reference strategy:** `{bp}` @ `{anchor}`. The harness measures it in iteration 1 on all "
        f"your splits (no need to put it in your slate); its accuracy is the floor and its cost per sample "
        f"the budget.",
        "- **How the run's final strategy is chosen:** every candidate is compared with the reference on the "
        "same questions of the blind split and then of the hidden split (never shown to you). Within 2 points "
        "of accuracy or 2% of cost is a tie. A candidate replaces the reference only with a WIN that holds on "
        "hidden too: **accuracy** more than 2 points higher at no more than +2% cost, or **cost** more than 2% "
        "lower with accuracy no more than 2 points lower. Otherwise the run reports the reference itself. "
        "Better-but-dearer or cheaper-but-worse never replaces it.",
        "- **Noise:** the same code re-run on the same rows changes cost by 5-20% and flips a few answers, and "
        "100 rows give accuracy ±3-4 points. Small gains are noise; a win has to be larger than that.",
    ]
    direction_only = blind_direction_only()
    rem = _remeasured(run_log, bp, anchor)
    if rem:
        blind = "" if direction_only else f"· blind acc={rem['bl_acc'] * 100:.1f}% "
        lines.append(
            f"- Reference measured in this run (iteration {rem['iteration']}): feedback acc={rem['fb_acc'] * 100:.1f}% "
            f"{blind}· cost/sample={rem['cost']:.0f}. Compare your candidates with these numbers.")
    else:
        lines.append("- The reference has not been measured yet; its numbers appear with the next results.")
    if direction_only:
        lines.append("- **Blind results** are shown to you only as a direction against the reference "
                     "(per candidate below and in the history).")
        return "\n".join(lines)
    vs = _vs_reference_blind(run_log, bp, anchor)
    if vs:
        lines.append("- **Your candidates vs the reference on the blind rows** (best 3, as the final pick ranks "
                     "them; the hidden check decides):")
        lines += vs
    else:
        lines.append("- **Your candidates vs the reference:** none measured yet.")
    return "\n".join(lines)
