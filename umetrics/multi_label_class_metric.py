# -*- coding: utf8 -*-
#
from collections import defaultdict
from typing import List, Dict

from abc import abstractmethod


class MultiLabelClassMacroF1Metric(object):
    def __init__(self):
        """
        计算多标签多分类macro f1
        """
        self.p = defaultdict(int)
        self.r = defaultdict(int)
        self.t = defaultdict(int)

    def step(self, preds: List[List[float]], labels: List[List[int]], threshold=0.5):

        for pred_item, label_item in zip(preds, labels):
            for index, p in enumerate(pred_item):
                if p > threshold:
                    self.p[index] += 1
            for index, r in enumerate(label_item):
                if r == 1:
                    self.r[index] += 1
            for index, (p, r) in enumerate(zip(pred_item, label_item)):
                if p > threshold and r == 1:
                    self.t[index] += 1

    def score(self) -> float:
        f1s = []
        for k, v in self.t.items():
            p_k = self.p[k]
            r_k = self.r[k]

            p = v / (p_k or 1e-5)
            r = v / (r_k or 1e-5)
            f1 = 2 * p * r / (p + r + 1e-5)
            f1s.append(f1)
        return sum(f1s) / (len(f1s) or 1e-5)

    _label_map = {}

    @property
    def label_map(self) -> Dict[str, int]:
        """
        提供label_map，可以做更精细的展示
        :return:
        """
        return self._label_map

    @label_map.setter
    def label_map(self, val: Dict[str, int]):
        self._label_map = val

    def report(self) -> Dict:
        id_label_map = {v: k for k, v in self.label_map.items()}

        for k, v in sorted(self.t.items(), key=lambda x: x[1]):
            k_name = id_label_map.get(k, k)
            p_k = self.p[k]
            r_k = self.r[k]

            p = v / (p_k or 1e-5)
            r = v / (r_k or 1e-5)
            f1 = 2 * p * r / (p + r + 1e-5)
            print(f'[{k_name}] count:{self.r[k]} p:{p} r:{r} f1:{f1}')

        return {}
