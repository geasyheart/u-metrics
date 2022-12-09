# -*- coding: utf8 -*-
#
from typing import List


class MicroMetrics(object):
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

    def f1_score(self):
        p_score = self.precision_score()
        r_score = self.recall_score()
        f1_score = 2 * (p_score * r_score) / ((p_score + r_score) or 1e-8)
        return f1_score

    def precision_score(self):
        r_count = sum(self.r_count.values())
        p_count = sum(self.p_count.values())
        p_score = r_count / (p_count or 1e-8)
        return p_score

    def recall_score(self):
        r_count = sum(self.r_count.values())
        t_count = sum(self.t_count.values())
        r_score = r_count / (t_count or 1e-8)
        return r_score
