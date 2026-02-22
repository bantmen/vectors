"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import recall_at_k, queries_per_second


class TestRecallAtK:
    def test_perfect_recall(self):
        pred = np.array([[0, 1, 2], [3, 4, 5]])
        gt = np.array([[0, 1, 2], [3, 4, 5]])
        assert recall_at_k(pred, gt) == 1.0

    def test_zero_recall(self):
        pred = np.array([[3, 4, 5], [6, 7, 8]])
        gt = np.array([[0, 1, 2], [0, 1, 2]])
        assert recall_at_k(pred, gt) == 0.0

    def test_partial_recall(self):
        pred = np.array([[0, 1, 5]])  # 2/3 correct
        gt = np.array([[0, 1, 2]])
        assert abs(recall_at_k(pred, gt) - 2 / 3) < 1e-6

    def test_recall_at_1(self):
        pred = np.array([[0, 5, 6], [3, 5, 6]])
        gt = np.array([[0, 1, 2], [3, 4, 5]])
        assert recall_at_k(pred, gt, k=1) == 1.0

    def test_order_independent(self):
        pred = np.array([[2, 0, 1]])
        gt = np.array([[0, 1, 2]])
        assert recall_at_k(pred, gt) == 1.0


class TestQPS:
    def test_basic(self):
        assert queries_per_second(100, 1.0) == 100.0

    def test_zero_time(self):
        assert queries_per_second(100, 0.0) == float("inf")
