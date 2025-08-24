"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class TreeNode:
    is_leaf: bool
    value: Optional[Any] = None
    feature: Optional[Union[str, int]] = None
    threshold: Optional[float] = None
    children: Optional[Dict[Any, "TreeNode"]] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]
    max_depth: int

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self._default_leaf = None
        self.tree: Optional[TreeNode] = None

    def _build_tree_dis_in_dis_out(self, X: pd.DataFrame, y: pd.Series, depth: int):
        if depth == self.max_depth or y.nunique() == 1 or X.shape[1] == 0:
            return TreeNode(is_leaf=True, value=y.mode()[0])
        best_attribute = opt_split_attribute(X, y, self.criterion)
        if best_attribute is None or best_attribute not in X.columns:
            return TreeNode(is_leaf=True, value=y.mode()[0])
        children = {}
        for value in X[best_attribute].unique():
            mask = (X[best_attribute] == value)
            subset_X = X.loc[mask].drop(columns=[best_attribute])
            subset_y = y.loc[mask]
            child = (
                self._build_tree_dis_in_dis_out(subset_X, subset_y, depth + 1)
                if len(subset_y) else TreeNode(is_leaf=True, value=y.mode()[0])
            )
            children[value] = child
        return TreeNode(is_leaf=False, feature=best_attribute, children=children)

    def _build_tree_real_in_dis_out(self, X: pd.DataFrame, y: pd.Series, depth: int):
        if depth == self.max_depth or y.nunique() == 1 or X.shape[1] == 0:
            return TreeNode(is_leaf=True, value=y.mode()[0])
        res = opt_split_attribute(X, y, self.criterion)
        if res is None:
            return TreeNode(is_leaf=True, value=y.mode()[0])
        best_attribute, best_threshold = res
        left_mask = X[best_attribute] <= best_threshold
        right_mask = ~left_mask
        left_X, left_y = X.loc[left_mask], y.loc[left_mask]
        right_X, right_y = X.loc[right_mask], y.loc[right_mask]
        left_child = (
            self._build_tree_real_in_dis_out(left_X, left_y, depth + 1)
            if len(left_y) else TreeNode(is_leaf=True, value=y.mode()[0])
        )
        right_child = (
            self._build_tree_real_in_dis_out(right_X, right_y, depth + 1)
            if len(right_y) else TreeNode(is_leaf=True, value=y.mode()[0])
        )
        return TreeNode(is_leaf=False, feature=best_attribute, threshold=float(best_threshold),
                        left=left_child, right=right_child)

    def _build_tree_dis_in_real_out(self, X: pd.DataFrame, y: pd.Series, depth: int):
        if depth == self.max_depth or y.nunique() == 1 or X.shape[1] == 0:
            return TreeNode(is_leaf=True, value=float(y.mean()))
        best_attribute = opt_split_attribute(X, y, self.criterion)
        if best_attribute is None or best_attribute not in X.columns:
            return TreeNode(is_leaf=True, value=float(y.mean()))
        children = {}
        for value in X[best_attribute].unique():
            mask = (X[best_attribute] == value)
            subset_X = X.loc[mask].drop(columns=[best_attribute])
            subset_y = y.loc[mask]
            child = (
                self._build_tree_dis_in_real_out(subset_X, subset_y, depth + 1)
                if len(subset_y) else TreeNode(is_leaf=True, value=float(y.mean()))
            )
            children[value] = child
        return TreeNode(is_leaf=False, feature=best_attribute, children=children)

    def _build_tree_real_in_real_out(self, X: pd.DataFrame, y: pd.Series, depth: int):
        if depth == self.max_depth or y.nunique() == 1 or X.shape[1] == 0:
            return TreeNode(is_leaf=True, value=float(y.mean()))
        res = opt_split_attribute(X, y, self.criterion)
        if res is None:
            return TreeNode(is_leaf=True, value=float(y.mean()))
        best_attribute, best_threshold = res
        left_mask = X[best_attribute] <= best_threshold
        right_mask = ~left_mask
        left_X, left_y = X.loc[left_mask], y.loc[left_mask]
        right_X, right_y = X.loc[right_mask], y.loc[right_mask]
        left_child = (
            self._build_tree_real_in_real_out(left_X, left_y, depth + 1)
            if len(left_y) else TreeNode(is_leaf=True, value=float(y.mean()))
        )
        right_child = (
            self._build_tree_real_in_real_out(right_X, right_y, depth + 1)
            if len(right_y) else TreeNode(is_leaf=True, value=float(y.mean()))
        )
        return TreeNode(is_leaf=False, feature=best_attribute, threshold=float(best_threshold),
                        left=left_child, right=right_child)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._default_leaf = float(y.mean()) if check_ifreal(y) else y.mode()[0]
        if (check_ifreal(X.iloc[:, 0]) == False and check_ifreal(y) == False):
            self.tree = self._build_tree_dis_in_dis_out(X, y, 0)
        elif (check_ifreal(X.iloc[:, 0]) == True and check_ifreal(y) == False):
            self.tree = self._build_tree_real_in_dis_out(X, y, 0)
        elif (check_ifreal(X.iloc[:, 0]) == False and check_ifreal(y) == True):
            self.tree = self._build_tree_dis_in_real_out(X, y, 0)
        else:
            self.tree = self._build_tree_real_in_real_out(X, y, 0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        def _coerce(name, index):
            if name in index:
                return name
            try:
                k = int(name)
                if k in index: return k
            except:
                pass
            try:
                k = float(name)
                if k in index: return k
            except:
                pass
            return name

        preds = []
        for _, row in X.iterrows():
            node = self.tree
            fallback = self._default_leaf
            while isinstance(node, TreeNode) and not node.is_leaf:
                if node.threshold is not None:
                    col_key = _coerce(node.feature, row.index)
                    node = node.left if row[col_key] <= node.threshold else node.right
                else:
                    col_key = _coerce(node.feature, row.index)
                    val = row[col_key]
                    child = node.children.get(val) if node.children is not None else None
                    if child is None:
                        node = TreeNode(is_leaf=True, value=fallback)
                        break
                    node = child
                    if node.is_leaf:
                        fallback = node.value
            preds.append(node.value if isinstance(node, TreeNode) and node.is_leaf else fallback)
        return pd.Series(preds)

    def plot(self) -> None:
        def print_tree(node: TreeNode, depth=0):
            if node.is_leaf:
                print("\t" * depth + f"Leaf: {node.value}")
                return
            if node.threshold is not None:
                print("\t" * depth + f"?({node.feature} <= {node.threshold})")
                print("\t" * (depth + 1) + "Yes:")
                print_tree(node.left, depth + 2)
                print("\t" * (depth + 1) + "No:")
                print_tree(node.right, depth + 2)
            else:
                print("\t" * depth + f"?({node.feature})")
                for v, child in (node.children or {}).items():
                    print("\t" * (depth + 1) + f"{v}:")
                    print_tree(child, depth + 2)

        print_tree(self.tree)
