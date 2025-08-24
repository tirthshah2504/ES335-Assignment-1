from typing import Union
import numpy as np
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    yh = pd.Series(y_hat).astype(object).to_numpy()
    yy = pd.Series(y).astype(object).to_numpy()
    mask = (~pd.isna(yh)) & (~pd.isna(yy))
    if mask.sum() == 0:
        return 0.0
    return float((yh[mask] == yy[mask]).mean())


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    yh = pd.Series(y_hat).astype(object).to_numpy()
    yy = pd.Series(y).astype(object).to_numpy()
    mask = (~pd.isna(yh)) & (~pd.isna(yy))
    if mask.sum() == 0:
        return 0.0
    pred_pos = (yh[mask] == cls)
    true_pos = (yy[mask] == cls)
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & (~true_pos)).sum())
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    yh = pd.Series(y_hat).astype(object).to_numpy()
    yy = pd.Series(y).astype(object).to_numpy()
    mask = (~pd.isna(yh)) & (~pd.isna(yy))
    if mask.sum() == 0:
        return 0.0
    pred_pos = (yh[mask] == cls)
    true_pos = (yy[mask] == cls)
    tp = int((pred_pos & true_pos).sum())
    fn = int(((~pred_pos) & true_pos).sum())
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    assert y_hat.size == y.size
    yh = pd.to_numeric(pd.Series(y_hat), errors="coerce")
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = (~yh.isna()) & (~yy.isna())
    if mask.sum() == 0:
        return 0.0
    diff = (yh[mask] - yy[mask]).to_numpy(dtype=float)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    yh = pd.to_numeric(pd.Series(y_hat), errors="coerce")
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = (~yh.isna()) & (~yy.isna())
    if mask.sum() == 0:
        return 0.0
    diff = (yh[mask] - yy[mask]).to_numpy(dtype=float)
    return float(np.mean(np.abs(diff)))
