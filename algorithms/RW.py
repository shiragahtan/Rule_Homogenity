from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from numpy.linalg import LinAlgError

# --- CONFIGURATION ---
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)
BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]
# How a walk is extended (see _homog_rw_direct). Default "weighted8" (instructor's decision): the original
# "top8" gave low-weight attributes probability 0, so violations that need them were never found.
#   "top8"            the 8 candidate predicates with the highest weights, greedy pick (original code)
#   "weighted8"       8 candidates drawn at random with probability proportional to their weight
#                     (every predicate has a chance > 0), then the same greedy pick
#   "weighted_single" one candidate drawn with probability proportional to its weight and added (article text)
RW_EXTENSION: str = _CFG.get("RW_EXTENSION", "weighted8")
sys.path.append(str(Path(__file__).resolve().parent.parent / "utils"))
from ATE_update import calculate_ate_safe


def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    parts: List[pd.DataFrame] = []
    lookup: Dict[str, Tuple[str, str]] = {}
    for col in df.columns:
        dummies = pd.get_dummies(df[col].astype(str), prefix=col, dtype=bool)
        parts.append(dummies)
        lookup.update({c: (col, c.split("_", 1)[1]) for c in dummies.columns})
    return pd.concat(parts, axis=1), lookup


def _homog_rw_direct(
        df: pd.DataFrame,
        *,
        treatment_col: str,
        outcome_col: str,
        delta: int,
        epsilon: float,
        ate_all: float,
        k_walks: int = 400,
        max_depth: int = 5,
        size_stop: float = 0.8,          # ← accepted, intentionally unused
        rng: Optional[random.Random] = None,
        attribute_weights: Optional[Dict[str, float]] = None,
        extension: Optional[str] = None,
        revisit: bool = False,
        arm_prefilter: bool = False,
        learned_weights: bool = False,
        learn_stats: Optional[Dict[str, List[float]]] = None,
        warm_start: bool = False,
        **_: object,                     # ← swallow future params safely
) -> Tuple[bool, int]:
    # Options tested for the Stack Overflow misses (all off by default = the behavior used in the paper so far):
    #   revisit        a walk may pass through a subgroup an earlier walk already checked (its effect is taken
    #                  from the cache and not counted again); without it, every checked subgroup blocks all
    #                  later walks, so deep regions below a common prefix are reached at most once.
    #   arm_prefilter  candidate predicates must keep both treatment arms >= delta/MIN_ARM_DELTA_DIVISOR rows
    #                  (the size filter of the estimator) before they are drawn; without it the 8 drawn
    #                  candidates can all fail the size filter and the walk ends early.
    #   max_depth      the number of predicates a walk may add after its first one (default 5).
    #   learned_weights  weights learned during the check instead of the config's ATTRIBUTE_WEIGHTS (proposal
    #                  of Sean and the instructor): every predicate starts at 1/n (n = number of predicates);
    #                  each time a predicate is added to a walk, its weight becomes 1/n + the mean, over the
    #                  times it was added, of |change of the subgroup's effect| / epsilon. Predicates that
    #                  move the effect more are drawn more often (roots too). Independent of the other options.
    #   learn_stats    LEARNING MODE (used by experiments/learn_rw_weights.py, never in a check): a dict that
    #                  collects, per attribute, [sum, count] of |change of the effect| / epsilon each time a
    #                  predicate of that attribute is added to a walk (the root: change from the overall effect).
    #                  The walks do not stop at a violation. weights_from_stats() turns the dict into the frozen,
    #                  normalized per-dataset weights that a check then gets as attribute_weights.
    #   warm_start     with learned_weights: start from attribute_weights (e.g. the frozen per-dataset weights)
    #                  instead of 1/n, and keep adapting during the check (base weight + mean change).


    rng = rng or random.Random()
    extension = extension or RW_EXTENSION
    if extension not in ("top8", "weighted8", "weighted_single"):
        raise ValueError(f"unknown RW extension {extension!r}")

    # ------------------ SETUP ------------------
    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")

    onehot, lookup = _onehot_lookup(mining_df)
    X = onehot.values
    Y = df[outcome_col].values
    T = df[treatment_col].values.astype(bool)
    _div = _CFG.get("MIN_ARM_DELTA_DIVISOR")
    min_arm = delta / _div if (arm_prefilter and delta and _div) else 0

    def arms_ok(m):
        if not min_arm:
            return True
        nt = int((m & T).sum())
        return nt >= min_arm and int(m.sum()) - nt >= min_arm
    col_names = list(onehot.columns)
    n_features = len(col_names)

    # ------------------ WEIGHTS (NO COMPRESSION) ------------------
    attr_w = attribute_weights or {}
    col_weights = np.array([
        attr_w.get(lookup[c][0], 1.0) for c in col_names
    ], dtype=float)
    if learned_weights:
        # same scale as the uniform start (sums to 1 over the predicates)
        base_weights = col_weights / col_weights.sum() if warm_start and col_weights.sum() > 0 \
            else np.full(n_features, 1.0 / max(n_features, 1))
        col_weights = base_weights.copy()
        change_sum = np.zeros(n_features)
        change_cnt = np.zeros(n_features)

    # ------------------ ROOT SCORING (RISK-BASED) ------------------
    root_scores = []
    root_indices = []
    root_diffs = []

    def learn(i, change):
        if learn_stats is not None and change is not None and np.isfinite(change) and epsilon:
            st = learn_stats.setdefault(lookup[col_names[i]][0], [0.0, 0])
            st[0] += abs(change) / epsilon
            st[1] += 1

    for i in range(n_features):
        mask = X[:, i]
        if mask.sum() < delta or not arms_ok(mask):
            continue

        try:
            sub_df = df[mask]
            cate = calculate_ate_safe(sub_df, treatment_col, outcome_col, delta)
            if not np.isfinite(cate):
                continue  # skip NaN/Inf
            diff = abs(cate - ate_all)
        except LinAlgError:
            continue

        score = col_weights[i] * (1.0 + diff * 10.0)
        if not np.isfinite(score):
            continue  # skip NaN/Inf
        if score <= 0.0:
            continue  # skip zero-weight

        root_scores.append(score)
        root_indices.append(i)
        root_diffs.append(diff)

    if not root_indices:
        return True, 0  # nothing valid to explore

    root_scores = np.array(root_scores, dtype=float)
    root_probs = root_scores / np.sum(root_scores)  # guaranteed finite now

    # ------------------ SEARCH ------------------
    visited: Set[frozenset] = set()
    cate_cache: Dict[frozenset, float] = {}
    checked = 0

    def eval_subset(mask, idxs):
        nonlocal checked
        key = frozenset(idxs)
        if key in visited:
            return cate_cache.get(key) if revisit else None
        visited.add(key)
        checked += 1

        if key in cate_cache:
            return cate_cache[key]

        try:
            val = calculate_ate_safe(df[mask], treatment_col, outcome_col, delta)
            cate_cache[key] = val
            return val
        except LinAlgError:
            return None

    # ------------------ WALKS ------------------
    chosen_roots = rng.choices(range(len(root_indices)), weights=root_probs, k=k_walks)
    root_diffs = np.array(root_diffs, dtype=float)

    for r in chosen_roots:
        if learned_weights:  # the root is drawn with the weights learned so far
            r = rng.choices(range(len(root_indices)),
                            weights=col_weights[root_indices] * (1.0 + root_diffs * 10.0))[0]
        start_idx = root_indices[r]
        mask = X[:, start_idx]
        idxs = {start_idx}
        attrs_used = {lookup[col_names[start_idx]][0]}

        cate = eval_subset(mask, idxs)
        cur_cate = cate if cate is not None else cate_cache.get(frozenset(idxs))
        if cur_cate is not None:
            learn(start_idx, cur_cate - ate_all)
        if learn_stats is None and cate is not None and abs(cate - ate_all) > epsilon:
            print(f"Breaking Subgroup: "
                  f"{ {lookup[col_names[i]][0]: lookup[col_names[i]][1] for i in idxs} }")
            return False, checked

        # ---------- GREEDY EXPANSION ----------
        for _ in range(max_depth):
            candidates = []

            for i in range(n_features):
                if i in idxs:
                    continue
                attr = lookup[col_names[i]][0]
                if attr in attrs_used:
                    continue

                new_mask = mask & X[:, i]
                if new_mask.sum() < delta or not arms_ok(new_mask):
                    continue

                candidates.append(i)

            if not candidates:
                break

            if extension == "top8":
                # evaluate top-K only (huge speedup)
                candidates = sorted(
                    candidates,
                    key=lambda i: col_weights[i],
                    reverse=True
                )[:8]
            else:
                # weighted sampling without replacement (Efraimidis-Spirakis keys u^(1/w)):
                # every candidate with a positive weight can be drawn
                n_draw = 1 if extension == "weighted_single" else 8
                pos = [i for i in candidates if col_weights[i] > 0]
                # log form of the key u^(1/w) (same order, no underflow for small learned weights)
                candidates = sorted(pos, key=lambda i: np.log(rng.random() or 1e-300) / col_weights[i],
                                    reverse=True)[:n_draw]

            best_gain = 0.0
            best_i = None
            best_mask = None
            best_cate = None

            for i in candidates:
                new_mask = mask & X[:, i]
                cate_new = eval_subset(new_mask, idxs | {i})
                if cate_new is None:
                    continue

                gain = abs(cate_new - ate_all)
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_mask = new_mask
                    best_cate = cate_new

            if best_i is None:
                break

            # commit step
            if learned_weights and cur_cate is not None and np.isfinite(cur_cate) and epsilon:
                change_sum[best_i] += abs(best_cate - cur_cate) / epsilon
                change_cnt[best_i] += 1
                col_weights[best_i] = base_weights[best_i] + change_sum[best_i] / change_cnt[best_i]
            if cur_cate is not None:
                learn(best_i, best_cate - cur_cate)
            cur_cate = best_cate
            mask = best_mask
            idxs.add(best_i)
            attrs_used.add(lookup[col_names[best_i]][0])

            if learn_stats is None and best_gain > epsilon:
                print(f"Breaking Subgroup: "
                      f"{ {lookup[col_names[i]][0]: lookup[col_names[i]][1] for i in idxs} }")
                return False, checked

    print(f"Total checked: {checked}")
    return True, checked


def weights_from_stats(stats: Dict[str, List[float]], attributes: List[str]) -> Dict[str, float]:
    """Frozen per-dataset attribute weights from the learning walks: weight = 1/n + mean(|effect change| / epsilon)
    over the times the attribute was added (0 if never added), normalized to sum to 1. The 1/n floor keeps every
    attribute drawable."""
    n = len(attributes)
    raw = {a: 1.0 / n + (stats[a][0] / stats[a][1] if a in stats and stats[a][1] else 0.0) for a in attributes}
    total = sum(raw.values())
    return {a: w / total for a, w in sorted(raw.items(), key=lambda kv: -kv[1])}



def calc_utility_for_subgroups(
        mode: int,
        algorithm: Callable,
        df: pd.DataFrame,
        treatment_col: str,
        delta: int,
        epsilon: float,
        *,
        outcome_col: Optional[str] = None,
        tgtO: Optional[str] = None,
        utility_all: Optional[float] = None,
        k_walks: int = 2000,  # article default (Parameter Rationale, section 6.1)
        size_stop: float = 0.8,
        rng: Optional[random.Random] = None,
        attribute_weights: Optional[Dict[str, float]] = None,
        **kwargs: object,
) -> Tuple[bool, int]:
    outcome_col = outcome_col or tgtO
    if outcome_col is None: raise ValueError("Need outcome_col")
    if utility_all is None: raise ValueError("Need utility_all for RW")
    if mode != 1:
        return _homog_rw_direct(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta,
            epsilon=epsilon,
            ate_all=utility_all,
            k_walks=k_walks,
            size_stop=size_stop,
            rng=rng,
            attribute_weights=attribute_weights,
            extension=kwargs.get("extension"),
            revisit=bool(kwargs.get("revisit", False)),
            arm_prefilter=bool(kwargs.get("arm_prefilter", False)),
            learned_weights=bool(kwargs.get("learned_weights", False)),
            learn_stats=kwargs.get("learn_stats"),
            warm_start=bool(kwargs.get("warm_start", False)),
            **({"max_depth": int(kwargs["max_depth"])} if kwargs.get("max_depth") else {}),
        )
    return [], 0