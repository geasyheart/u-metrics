# -*- coding: utf8 -*-
#

from umetrics.corefmetrics import CorefEvaluator

evaluator = CorefEvaluator()

A = (1, 3)
B = (4, 4)
C = (6, 7)
D = (9, 10)

gold_clusters = [
    (A, B, C, D),

]
mention_to_gold = {}
for gc in gold_clusters:
    for mention in gc:
        mention_to_gold[tuple(mention)] = gc

predicted_clusters = [
    (A, B, C)

]

mention_to_predicted = {}
for gc in predicted_clusters:
    for mention in gc:
        mention_to_predicted[tuple(mention)] = gc

evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

print(evaluator.get_prf())
