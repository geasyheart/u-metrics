# -*- coding: utf8 -*-
#
import random
from unittest import TestCase

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from macrometrics import MacroMetrics


class TestMacroMetrics(TestCase):
    def test_aa(self):
        m = MacroMetrics(labels=[1, 2])
        m.step(y_preds=[0, 0, 0, 0, 1, 1, 1, 2, 2],
               y_trues=[0, 0, 1, 2, 1, 1, 2, 1, 2])
        m.classification_report()

        print(classification_report([0, 0, 1, 2, 1, 1, 2, 1, 2], [0, 0, 0, 0, 1, 1, 1, 2, 2], zero_division=0, labels=[1, 2]))

    def test_a(self):
        y_preds = [1, 1, 2, 3, 2, 2, 3, 2, 3, 0, 0]
        y_trues = [1, 1, 1, 1, 2, 2, 2, 3, 3, 0, 0]

        labels = list(set(y_preds) & set(y_trues))

        m = MacroMetrics(labels=labels)
        m.step(y_trues=y_trues, y_preds=y_preds)
        self.assertEqual(
            round(precision_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro'), 5),
            round(m.precision_score(), 5)
        )
        self.assertEqual(
            round(recall_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro')),
            round(m.recall_score())
        )
        self.assertEqual(
            round(f1_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro')),
            round(m.f1_score())
        )

        print(classification_report(y_trues, y_preds, zero_division=0, labels=labels))
        m.classification_report()

    def test_b(self):
        y_trues = [random.randint(0, 10) for i in range(10000)]
        y_preds = [random.randint(0, 10) for i in range(10000)]

        labels = list(set(y_preds) & set(y_trues))

        m = MacroMetrics(labels=labels)
        m.step(y_trues=y_trues, y_preds=y_preds)
        self.assertEqual(
            round(precision_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro'), 5),
            round(m.precision_score(), 5)
        )
        self.assertEqual(
            round(recall_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro')),
            round(m.recall_score())
        )
        self.assertEqual(
            round(f1_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro')),
            round(m.f1_score())
        )

        print(classification_report(y_trues, y_preds, zero_division=0, labels=labels))

        m.classification_report()

    def test_c(self):
        y_trues = [random.randint(0, 10) for i in range(10000)]
        y_preds = [random.randint(0, 10) for i in range(10000)]

        labels = list(set(y_preds) & set(y_trues))

        p_score = round(precision_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro'), 5)
        r_score = round(recall_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro'), 5)
        f_score = round(f1_score(y_trues, y_preds, labels=labels, zero_division=0, average='macro'), 5)

        y_trues_chunk = [y_trues[i:i + 3] for i in range(0, len(y_trues), 3)]
        y_preds_chunk = [y_preds[i:i + 3] for i in range(0, len(y_preds), 3)]

        m = MacroMetrics(labels=labels)
        for y_true_chunk, y_pred_chunk in zip(y_trues_chunk, y_preds_chunk):
            m.step(y_trues=y_true_chunk, y_preds=y_pred_chunk)

        self.assertEqual(p_score, round(m.precision_score(), 5))
        self.assertEqual(r_score, round(m.recall_score(), 5))
        self.assertEqual(f_score, round(m.f1_score(), 5))
