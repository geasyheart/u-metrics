# -*- coding: utf8 -*-
#
from typing import List


class MacroMetrics(object):
    def __init__(self, labels: List[int]):
        self.labels = labels

        self.p_count = {key: 0 for key in labels}
        self.r_count = {key: 0 for key in labels}
        self.t_count = {key: 0 for key in labels}

    def step(self, y_trues: List[int], y_preds: List[int]):
        assert len(y_trues) == len(y_preds)

        # for y_true, y_pred in zip(y_trues, y_preds):
        for i in range(len(y_trues)):
            y_true = y_trues[i]
            y_pred = y_preds[i]
            try:
                self.p_count[y_pred] += 1
            except KeyError:
                pass
            try:
                self.t_count[y_true] += 1
            except KeyError:
                pass
            if y_true == y_pred:
                try:
                    self.r_count[y_true] += 1
                except KeyError:
                    pass

    def precision_score(self):
        scores = 0.
        for label in self.labels:
            r_c = self.r_count[label]
            p_c = self.p_count[label]
            label_score = r_c / (p_c or 1e-8)
            scores += label_score
        return scores / (len(self.labels) or 1e-8)

    def recall_score(self):
        scores = 0.

        for label in self.labels:
            r_c = self.r_count[label]
            t_c = self.t_count[label]
            label_score = r_c / (t_c or 1e-8)
            scores += label_score
        return scores / (len(self.labels) or 1e-8)

    def f1_score(self):
        f1_scores = 0.
        for label in self.labels:
            r_c = self.r_count[label]
            p_c = self.p_count[label]
            t_c = self.t_count[label]
            p_score = r_c / (p_c or 1e-8)
            r_score = r_c / (t_c or 1e-8)

            f1_score = 2 * (p_score * r_score) / ((p_score + r_score) or 1e-8)
            f1_scores += f1_score
        return f1_scores / (len(self.labels) or 1e-8)

    def classification_report(self, out_func=print):
        label_scores = {label: {} for label in self.labels}
        for label in self.labels:
            r_c = self.r_count[label]
            p_c = self.p_count[label]
            t_c = self.t_count[label]
            p_score = r_c / (p_c or 1e-8)
            r_score = r_c / (t_c or 1e-8)

            f1_score = 2 * (p_score * r_score) / ((p_score + r_score) or 1e-8)
            label_scores[label].setdefault('p_score', p_score)
            label_scores[label].setdefault('r_score', r_score)
            label_scores[label].setdefault('f1_score', f1_score)
            label_scores[label].setdefault('support', t_c)
        # to_table
        out_func("{:<10} {:<20} {:<20} {:<20} {:<20}".format('label', 'precision', 'recall', 'f1-score', 'support'))
        for label, scores in label_scores.items():
            out_func("{:<10} {:<20} {:<20} {:<20} {:<20}".format(
                label,
                scores['p_score'],
                scores['r_score'],
                scores['f1_score'],
                scores['support'],
            ))
