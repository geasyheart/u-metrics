# -*- coding: utf8 -*-
#
from typing import List, Tuple, Dict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class BLEUMetric(object):
    def __init__(self):
        self.chencherry = SmoothingFunction()
        self.ng1 = 0
        self.ng2 = 0
        self.ng3 = 0
        self.ng4 = 0
        self.avg_ng = 0

        self.count = 0

    def step(self, y_trues: List[Tuple[str]], y_preds: List[List[Tuple[str]]]):
        assert len(y_trues) == len(y_preds)
        for y_true, y_pred in zip(y_trues, y_preds):
            self.ng1 += sentence_bleu(y_pred, y_true, weights=(1, 0, 0, 0), smoothing_function=self.chencherry.method1)
            self.ng2 += sentence_bleu(y_pred, y_true, weights=(0, 1, 0, 0), smoothing_function=self.chencherry.method1)
            self.ng3 += sentence_bleu(y_pred, y_true, weights=(0, 0, 1, 0), smoothing_function=self.chencherry.method1)
            self.ng4 += sentence_bleu(y_pred, y_true, weights=(0, 0, 0, 1), smoothing_function=self.chencherry.method1)
            self.avg_ng += sentence_bleu(y_pred, y_true, weights=(0.25, 0.25, 0.25, 0.25),
                                         smoothing_function=self.chencherry.method1)

            self.count += 1

    def score(self):
        return self.avg_ng / (self.count + 1e-5)

    def report(self) -> Dict[str, float]:
        return {
            "bleu1": self.ng1/(self.count+1e-5),
            "bleu2": self.ng2/(self.count+1e-5),
            "bleu3": self.ng3/(self.count+1e-5),
            "bleu4": self.ng4/(self.count+1e-5),
            "bleuAvg": self.avg_ng/(self.count+1e-5),
        }


if __name__ == '__main__':
    b = BLEUMetric()
    b.step(
        [('there', 'is', 'a', 'cat', 'on', 'the', 'table')],
        [[('a', 'cat', 'is', 'on', 'the', 'table')]],
    )

    print(b.report())