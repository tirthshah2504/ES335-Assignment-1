"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these functions are here to simply help you.
"""

import numpy as np
import pandas as pd


# ---------------------------
# Encoding
# ---------------------------

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding on multi-category discrete columns.
    Leaves binary discrete columns as-is.
    """
    for column in X.columns:
        if not check_ifreal(X[column]) and X[column].nunique(dropna=False) > 2:
            dummies = pd.get_dummies(X[column], prefix=column)
            X = pd.concat([X, dummies], axis=1)
            X.drop(column, axis=1, inplace=True)
    return X


# ---------------------------
# Real vs Discrete detection
# ---------------------------

def check_ifreal(y: pd.Series, real_distinct_threshold: int = 6) -> bool:
    """
    Heuristic to decide if a series is 'real/continuous' or 'discrete'.
    - categorical/bool/string -> discrete
    - float -> real
    - int -> real if #distinct >= threshold, else discrete
    """
    if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y) or pd.api.types.is_string_dtype(y):
        return False
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_integer_dtype(y):
        return int(pd.Series(y).nunique(dropna=False)) >= real_distinct_threshold
    return False


# ---------------------------
# Impurity / Error metrics
# ---------------------------

def entropy(Y: pd.Series) -> float:
    """
    Entropy = -sum p_i log2 p_i
    """
    if len(Y) == 0:
        return 0.0
    counts = Y.value_counts(normalize=True, dropna=False).to_numpy(dtype=float)
    # add tiny epsilon for numerical stability
    return float(-np.sum(counts * np.log2(counts + 1e-12)))


def gini_index(Y: pd.Series) -> float:
    """
    Gini = 1 - sum p_i^2
    """
    if len(Y) == 0:
        return 0.0
    p = Y.value_counts(normalize=True, dropna=False).to_numpy(dtype=float)
    return float(1.0 - np.sum(p * p))


def mse(Y: pd.Series) -> float:
    """
    Mean Squared Error around the mean (population version)
    """
    if len(Y) == 0:
        return 0.0
    mu = float(np.mean(Y))
    return float(np.mean((Y - mu) ** 2))


# ---------------------------
# Criterion routing
# ---------------------------

def check_criteria(Y: pd.Series, criterion: str):
    """
    Map (Y type, criterion name) -> concrete function ('entropy' | 'gini_index' | 'mse').
    - If criterion == "entropy":
        * classification (discrete Y): entropy
        * regression     (real Y):     mse (i.e., variance reduction)
    - If criterion == "gini_index": use gini for classification; for real Y it’s uncommon,
      but we keep gini for compatibility with the template.
    """
    if criterion == "entropy":
        this_criteria = "mse" if check_ifreal(Y) else "entropy"
    elif criterion == "gini_index":
        this_criteria = "gini_index"
    else:
        # default safe fallback (use entropy for classification, mse for regression)
        this_criteria = "mse" if check_ifreal(Y) else "entropy"

    func_map = {
        "entropy": entropy,
        "gini_index": gini_index,
        "mse": mse,
    }
    return this_criteria, func_map[this_criteria]


# ---------------------------
# Threshold search (numeric features)
# ---------------------------

def opt_threshold(Y: pd.Series, attr: pd.Series, criterion: str):
    """
    Find optimal threshold for a numeric feature by scanning midpoints of sorted unique values.
    Returns the threshold (float) or None if not possible.
    """
    # ensure numeric attribute
    if not check_ifreal(attr):
        return None

    this_criteria, crit_func = check_criteria(Y, criterion)

    xs = pd.Series(attr, copy=False).astype(float).sort_values().to_numpy()
    if xs.size <= 1:
        return None

    uniq = np.unique(xs)
    if uniq.size <= 1:
        return None

    # candidate thresholds are midpoints between unique consecutive values
    mids = (uniq[:-1] + uniq[1:]) / 2.0

    best_thr = None
    best_gain = -np.inf

    for thr in mids:
        left_mask = attr <= thr
        Y_left = Y[left_mask]
        Y_right = Y[~left_mask]
        if len(Y_left) == 0 or len(Y_right) == 0:
            continue

        # weighted child impurity
        w_left = len(Y_left) / len(Y)
        w_right = 1.0 - w_left
        child = w_left * crit_func(Y_left) + w_right * crit_func(Y_right)
        gain = crit_func(Y) - child

        if gain > best_gain:
            best_gain = gain
            best_thr = float(thr)

    return best_thr


# ---------------------------
# Information gain wrapper
# ---------------------------

def information_gain(Y: pd.Series, attribute: pd.Series, criterion: str = None) -> float:
    """
    IG = crit(Y) - sum_i (|Y_i| / |Y|) * crit(Y_i)
    - For numeric attributes: split at best threshold (via opt_threshold)
    - For discrete attributes: group by unique values
    """
    this_criteria, crit_func = check_criteria(Y, criterion)

    # numeric attribute -> threshold split
    if check_ifreal(attribute):
        thr = opt_threshold(Y, attribute, criterion)
        if thr is None:
            return 0.0
        Y_left = Y[attribute <= thr]
        Y_right = Y[attribute > thr]
        w_left = len(Y_left) / len(Y)
        w_right = 1.0 - w_left
        return crit_func(Y) - (w_left * crit_func(Y_left) + w_right * crit_func(Y_right))

    # discrete attribute -> multiway split
    total = 0.0
    n = len(Y)
    if n == 0:
        return 0.0
    for v in attribute.dropna().unique():
        Y_i = Y[attribute == v]
        if len(Y_i) == 0:
            continue
        total += (len(Y_i) / n) * crit_func(Y_i)
    return crit_func(Y) - total


# ---------------------------
# Best attribute selection
# ---------------------------

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features: pd.Series, criterion: str):
    """
    From 'features' choose the attribute with maximum information gain
    (entropy/Gini for classification, MSE for regression when criterion='information_gain').
    """
    best_feature = None
    best_gain = -np.inf

    for feature in features:
        try:
            gain = information_gain(y, X[feature], criterion)
        except Exception:
            continue
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature


# ---------------------------
# Data splitting helpers
# ---------------------------

def split_data_discrete(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Split for discrete attribute:
      left:  X[attr] == value
      right: X[attr] != value
    Returns (X_left, y_left, X_right, y_right)
    """
    mask_left = (X[attribute] == value)
    X_left = X.loc[mask_left]
    X_right = X.loc[~mask_left]
    y_left = y.loc[X_left.index]
    y_right = y.loc[X_right.index]
    return X_left, y_left, X_right, y_right


def split_data_real(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Split for numeric attribute:
      left:  X[attr] <= value
      right: X[attr] >  value
    Returns (X_left, y_left, X_right, y_right)
    """
    mask_left = (X[attribute] <= value)
    X_left = X.loc[mask_left]
    X_right = X.loc[~mask_left]
    y_left = y.loc[X_left.index]
    y_right = y.loc[X_right.index]
    return X_left, y_left, X_right, y_right


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Generic split that dispatches to discrete/real version.
    Always returns (X_left, y_left, X_right, y_right).
    """
    if check_ifreal(X[attribute]):
        return split_data_real(X, y, attribute, value)
    else:
        return split_data_discrete(X, y, attribute, value)
