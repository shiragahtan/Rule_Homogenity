from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

# --- CONFIGURATION ---
# Load config to get Treatment Column
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]

# Import ATE calculation helper
sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe


# helper function: One-Hot Encoding & Lookup
def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Preprocesses the data into One-Hot columns.
    Returns:
        1. DataFrame of booleans (The Matrix)
        2. Dictionary mapping {OneHotCol -> OriginalAttribute}
        3. List of column names for indexing
    """
    parts: List[pd.DataFrame] = []
    lookup: Dict[str, str] = {}

    for col in df.columns:
        # Create dummies, treat NAs as a distinct category
        dummies = pd.get_dummies(df[col].fillna("⧫NA⧫").astype(str), prefix=col, dtype=bool)
        parts.append(dummies)

        # Map specific column "Age_18" back to attribute "Age"
        for c in dummies.columns:
            lookup[c] = col

    final_df = pd.concat(parts, axis=1)
    return final_df, lookup, list(final_df.columns)


# CORE ALGORITHM: Best-First Search (Max-Heap)
def _best_first_subgroup_search(
        df: pd.DataFrame,
        *,
        treatment_col: str,
        outcome_col: str,
        delta: int,
        epsilon: float,
        utility_all: float,
) -> Tuple[bool, int]:  # <--- CHANGED RETURN TYPE
    """
    Best-First Search prioritized by Subgroup Size.

    Strategy:
    1. Maintain a Max-Heap of subgroups, ordered by size (largest first).
    2. Pop the largest subgroup.
    3. Check homogeneity (ATE check). If violated, return False immediately.
    4. "Expand" this subgroup: Generate all 'children' by adding 1 new filter.
       - Only add children if their size >= delta.
       - Only add children if we haven't processed that combination of filters yet.
    5. Repeat until the heap is empty or the largest item in the heap < delta.

    Returns:
        (is_homogeneous: bool, number_of_subgroups_checked: int)
    """

    # 1. Prepare Data
    # Drop columns we shouldn't filter on
    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")

    # Create the Boolean Matrix (Rows x Features)
    onehot_df, col_to_attr, col_names = _onehot_lookup(mining_df)
    X_matrix = onehot_df.values
    n_features = X_matrix.shape[1]
    total_rows = len(df)

    # 2. Heap Initialization
    # Python's heapq is a Min-Heap. To simulate Max-Heap, we store (-size).
    # Structure: (-size, [list_of_feature_indices], {set_of_used_attributes})
    # We start with the "Root" (Empty filters, meaning all rows).

    # Root represents the full dataset
    initial_indices: List[int] = []
    initial_used_attrs: Set[str] = set()

    # Heap stores: (priority_neg_size, sort_key_tuple, feature_indices_list, used_attrs_set)
    # We need 'sort_key_tuple' (tuple of indices) because lists aren't comparable in ties.
    pq = []
    heapq.heappush(pq, (-total_rows, tuple(), initial_indices, initial_used_attrs))

    # Visited set to avoid redundant processing (e.g., {A, B} vs {B, A})
    # Stores tuple(sorted(feature_indices))
    visited: Set[Tuple[int, ...]] = set()
    visited.add(tuple())

    # Counter for checked subgroups
    checked_count = 0

    while pq:
        # --- A. POP LARGEST SUBGROUP ---
        neg_size, _, current_indices, current_used_attrs = heapq.heappop(pq)
        current_size = -neg_size

        # If the largest remaining group is smaller than delta, we are done.
        # Since it's a Max-Heap, everything else is also smaller.
        if current_size < delta:
            print("Terminating: Largest remaining subgroup in heap is smaller than delta.")
            break

        # We are about to "check" this node (calculate ATE or attempt to)
        checked_count += 1

        # Reconstruct Mask
        # (We don't store masks in heap to save memory, we reconstruct via fast bitwise AND)
        if not current_indices:
            current_mask = np.ones(total_rows, dtype=bool)
        else:
            # Start with the first feature's column
            current_mask = X_matrix[:, current_indices[0]].copy()
            # AND with the rest
            for idx in current_indices[1:]:
                current_mask &= X_matrix[:, idx]

        # --- B. CHECK HOMOGENEITY (The "Validation") ---
        # Construct path name for logging
        path_names = [col_names[i] for i in current_indices]

        # >>>> DEBUG PRINT <<<<
        # print(f"Checking Subgroup: {path_names} | Size: {current_size}")

        try:
            sub_df = df[current_mask]

            # Calculate Utility (ATE)
            ate_sub = calculate_ate_safe(sub_df, treatment_col, outcome_col, delta)

            # Compare with Global Utility
            diff = abs(ate_sub - utility_all)

            if diff > epsilon:
                print(f"  >>> VIOLATION FOUND! Path={path_names}")
                print(
                    f"  >>> Size: {current_size}, ATE Sub: {ate_sub:.4f}, ATE All: {utility_all:.4f}, Diff: {diff:.4f}")
                print(f"Total unique subgroups checked: {checked_count}")
                return False, checked_count

        except LinAlgError:
            # Singular matrix errors still count as a "check" attempt
            print(f"  > LinAlgError for path {path_names}. Skipping check, but will try children.")
            pass

        # --- C. EXPAND (Generate Children) ---
        # Try adding every possible unused attribute

        # Optimization: We only calculate the size of children.
        # We do NOT check their ATE yet. That happens only when they are popped.

        # Filter candidate columns:
        # 1. Attribute not already used (e.g., don't filter 'Age' if 'Age' is present)
        candidate_indices = [
            i for i in range(n_features)
            if col_to_attr[col_names[i]] not in current_used_attrs
        ]

        if not candidate_indices:
            continue

        # Get the sub-matrix for the current rows
        # shape: (current_size, n_features)
        # We only care about columns in candidate_indices
        subset_matrix = X_matrix[current_mask]

        # Vectorized Size Check:
        # Summing columns of the subset gives the size of the child node immediately
        child_sizes = subset_matrix[:, candidate_indices].sum(axis=0)

        for i, idx in enumerate(candidate_indices):
            new_size = child_sizes[i]

            # 1. Pruning: Size < Delta
            if new_size < delta:
                continue

            # 2. Create New State
            new_indices = current_indices + [idx]
            # Sort to ensure uniqueness in 'visited'
            new_key = tuple(sorted(new_indices))

            # 3. Pruning: Already Visited
            if new_key in visited:
                continue

            visited.add(new_key)

            # 4. Update Attributes Used
            new_attr = col_to_attr[col_names[idx]]
            new_used_attrs = current_used_attrs.copy()
            new_used_attrs.add(new_attr)

            # 5. Push to Heap
            # Priority is (-size). Tie-breaker is new_key (tuple), which is comparable.
            heapq.heappush(pq, (-new_size, new_key, new_indices, new_used_attrs))

    # If we exit the loop without returning False, no heterogeneity was found
    print(f"Total unique subgroups checked: {checked_count}")
    return True, checked_count


# -------------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------------
def calc_utility_for_subgroups(
        mode: int,
        df: pd.DataFrame,
        treatment_col: str,
        delta: int,
        epsilon: float,
        utility_all: float,
        *,
        tgtO: Optional[str] = None,
        **kwargs: object,
) -> Tuple[bool, int]:  # <--- UPDATED SIGNATURE
    """
    Dispatcher.
    Mode 0: Runs Best-First Search (Prioritized by Size) to detect heterogeneity.
    """
    if mode == 0:
        return _best_first_subgroup_search(
            df,
            treatment_col=treatment_col,
            outcome_col=tgtO,
            delta=delta,
            epsilon=epsilon,
            utility_all=utility_all
        )

    # Default return for other modes or if needed
    return [], 0