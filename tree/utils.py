"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
from typing import Optional, Tuple, Union, List
import numpy as np
def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    for col in X.select_dtypes(include=['category', 'object']).columns:
        dummies = pd.get_dummies(X[col], prefix=col)
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(columns=[col])
    return X
    

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """

    if y.dtype in [int, float] or pd.api.types.is_float_dtype(y) or pd.api.types.is_integer_dtype(y):
        return True
    else:
        return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    val = Y.value_counts(normalize=True)
    ent = 0
    for p in val:
        if p > 0:
            ent += -p * np.log2(p)
    return ent

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """

    val = Y.value_counts(normalize=True)
    gini = 1
    for p in val:
        gini -= p**2
    return gini

def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """

    return ((Y - Y.mean()) ** 2).mean()

def variance_reduction(Y: pd.Series, parts: List[pd.Series]) -> float:
    """
    Function to calculate the variance reduction
    """

    base_variance = Y.var()
    total_count = len(Y)
    weighted_variance = 0
    for part in parts:
        weighted_variance += (len(part) / total_count) * part.var()
    var_reduction = base_variance - weighted_variance
    return var_reduction

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    if criterion == 'entropy':
        base_entropy = entropy(Y)
        val = attr.value_counts(normalize=True)
        cond_entropy = 0
        for v, p in val.items():
            cond_entropy += p * entropy(Y[attr == v])
        info_gain = base_entropy - cond_entropy
        return info_gain

    elif criterion == 'gini_index':
        base_gini = gini_index(Y)
        val = attr.value_counts(normalize=True)
        cond_gini = 0
        for v, p in val.items():
            cond_gini += p * gini_index(Y[attr == v])
        info_gain = base_gini - cond_gini
        return info_gain

    elif criterion == 'MSE':
        base_mse = ((Y - Y.mean()) ** 2).mean()
        val = attr.value_counts(normalize=True)
        cond_mse = 0
        for v, p in val.items():
            cond_mse += p * ((Y[attr == v] - Y[attr == v].mean()) ** 2).mean()
        info_gain = base_mse - cond_mse
        return info_gain

    else:
        raise ValueError("Invalid criterion. Choose from 'entropy', 'gini_index', or 'MSE'.")

def information_gain_from_parts(Y: pd.Series, parts: List[pd.Series]) -> float:
    H_parent = entropy(Y)
    N = len(Y)
    if N == 0:
        return 0.0
    return H_parent - sum((len(p) / N) * entropy(p) for p in parts if len(p))

def gini_decrease_from_parts(Y: pd.Series, parts: List[pd.Series]) -> float:
    G_parent = gini_index(Y)
    N = len(Y)
    if N == 0:
        return 0.0
    return G_parent - sum((len(p) / N) * gini_index(p) for p in parts if len(p))

def opt_split_attribute(
    X: pd.DataFrame,
    y: pd.Series,
    criterion: str = "information_gain",
)-> Optional[Union[str, Tuple[str, float]]]:
    """
    Choose the best split.

    Returns:
      - str (attribute name) for DISCRETE-X splits (classification or regression)
      - (str, float) for REAL-X splits (classification or regression)
      - None if no valid split found.

    For classification (y discrete):
        criterion: "information_gain" | "gini_index"
    For regression  (y real):
        criterion is ignored; we maximize variance reduction.
    """
    if X is None or X.shape[1] == 0 or len(y) == 0 or y.nunique(dropna=False) <= 1:
        return None

    y_is_real = check_ifreal(y)
    real_cols = [c for c in X.columns if check_ifreal(X[c])]
    cat_cols  = [c for c in X.columns if not check_ifreal(X[c])]

    if not y_is_real:
        scorer = gini_decrease_from_parts if criterion == "gini_index" else information_gain_from_parts
    else:
        scorer = variance_reduction

    best_score = -np.inf
    best_attr: Optional[str] = None
    best_pair: Optional[Tuple[str, float]] = None

    for col in real_cols:
        x = X[col].to_numpy(dtype=float)
        order = np.argsort(x)
        x_sorted = x[order]
        idx = np.where(np.diff(x_sorted) != 0)[0]
        if idx.size == 0:
            continue
        for i in idx:
            thr = (x_sorted[i] + x_sorted[i + 1]) / 2.0
            left_mask = X[col] <= thr
            yL, yR = y[left_mask], y[~left_mask]
            if len(yL) == 0 or len(yR) == 0:
                continue
            score = scorer(y, [yL, yR])
            if score > best_score:
                best_score = score
                best_pair = (col, float(thr))
                best_attr = None

    for col in cat_cols:
        parts = [y[X[col] == v] for v in X[col].dropna().unique()]
        if len(parts) <= 1:
            continue
        score = scorer(y, parts)
        if score > best_score:
            best_score = score
            best_attr = col
            best_pair = None

    if best_pair is not None:
        return best_pair
    if best_attr is not None:
        return best_attr
    return None

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """
    if (check_ifreal(X[attribute])==False):
        subset_X = X[X[attribute] == value].drop(columns=[attribute])
        subset_y = y[X[attribute] == value]
        return subset_X, subset_y
    else:  
        subset_X_left = X[X[attribute] <= value]
        subset_X_right = X[X[attribute] > value]
        y_left = y[X[attribute] <= value]
        y_right = y[X[attribute] > value]
        return subset_X_left, subset_X_right, y_left, y_right
