"""
Simplified Decision Tree (same class shape as the reference code)

Supports:
- discrete input, discrete output
- real input, real output
- real input, discrete output
- discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd
from tree.utils import *  # expects: check_ifreal, opt_split_attribute, opt_threshold, split_data, information_gain

np.random.seed(42)


@dataclass
class Node:
    
    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=False, output=None, gain=0.0):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.output = output
        self.gain = gain

    def check_leaf(self):
        return self.is_leaf


class DecisionTree:
    criterion: Literal["entropy", "gini_index"]  # criterion won't be used for regression
    max_depth: int

    def __init__(self, criterion: str, max_depth: int = 10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root_node: Union[Node, None] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> None:
        """
        Train and construct the decision tree.
        Handles real/discrete input and real/discrete output automatically.
        """

        def make_leaf(y_: pd.Series) -> Node:
            if check_ifreal(y_):
                return Node(is_leaf=True, output=float(np.round(y_.mean(), 4)))
            else:
                return Node(is_leaf=True, output=y_.mode()[0])

        def build(X_: pd.DataFrame, y_: pd.Series, d: int) -> Node:
            # stopping conditions
            if d >= self.max_depth or y_.nunique() <= 1 or X_.shape[1] == 0:
                return make_leaf(y_)

            # choose best attribute
            best_attr = opt_split_attribute(X_, y_, X_.columns, self.criterion)
            if best_attr is None or best_attr not in X_.columns:
                return make_leaf(y_)

            # choose split "value"
            if check_ifreal(X_[best_attr]):
                split_val = opt_threshold(y_, X_[best_attr], self.criterion)
            else:
                # for discrete attr, split on most informative category (or majority as a fallback)
                split_val = X_[best_attr].mode()[0]

            # perform split
            X_left, y_left, X_right, y_right = split_data(X_, y_, best_attr, split_val)

            # if split failed, back off to leaf
            if len(y_left) == 0 or len(y_right) == 0:
                return make_leaf(y_)

            # compute split gain (optional metadata)
            try:
                best_gain = float(information_gain(y_, X_[best_attr], self.criterion))
            except Exception:
                best_gain = 0.0

            # recurse
            left = build(X_left, y_left, d + 1)
            right = build(X_right, y_right, d + 1)

            return Node(attribute=best_attr, value=split_val, left=left, right=right, is_leaf=False, gain=best_gain)

        self.root_node = build(X, y, depth)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict outputs for rows in X by traversing the trained tree.
        """

        def is_numeric(v) -> bool:
            return isinstance(v, (int, float, np.integer, np.floating))

        def predict_row(row: pd.Series) -> Union[float, int, str]:
            node = self.root_node
            while node is not None and not node.check_leaf():
                attr = node.attribute
                val = node.value
                xval = row[attr]

                # Decide split type by the stored split value:
                # numeric 'value' => threshold split; else equality split
                if is_numeric(val):
                    node = node.left if xval <= val else node.right
                else:
                    node = node.left if xval == val else node.right

            # leaf
            return node.output if node is not None else None

        return pd.Series([predict_row(x) for _, x in X.iterrows()])

    def plot(self, path: str = None) -> None:
        """
        Pretty-print tree in text form.
        """
        if not self.root_node:
            print("Tree not trained yet")
            return

        print("\nTree Structure:")
        print(self.print_tree())

    def print_tree(self) -> str:
        def is_numeric(v) -> bool:
            return isinstance(v, (int, float, np.integer, np.floating))

        def fmt_split(attr, val) -> str:
            if is_numeric(val):
                return f'?(attribute {attr} <= {val:.4f})'
            return f'?(attribute {attr} == {val})'

        def print_node(node: Node, indent: str = '') -> str:
            if node.is_leaf:
                return f'Class {node.output}\n'
            s = f'{fmt_split(node.attribute, node.value)}\n'
            s += f'{indent}Y: ' + print_node(node.left, indent + '    ')
            s += f'{indent}N: ' + print_node(node.right, indent + '    ')
            return s

        return print_node(self.root_node)
