## Example

```python
# macro
from umetrics import MacroMetrics
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

import random

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

assert p_score == round(m.precision_score(), 5)
assert r_score == round(m.recall_score(), 5)
assert f_score == round(m.f1_score(), 5)

```


## The Package Mission

Compared with sklearn already provides good and mature functions, For example:
1) precision_score
2) recall_score
3) f1_score
4) classification_report

Why write such a project?

For example in forecasting:

```python
from sklearn.metrics import precision_score
y_true = []
y_pred = []
precision_score(y_true=y_true,y_pred=y_pred)
```

Assuming that the amount of data in `y_true` or `y_pred` is very large, then just storing these data will already consume a lot of memory, let alone calculations.


## 此包存在的目的

相比sklearn已经提供好的并且很成熟的函数，例如:
1) precision_score
2) recall_score
3) f1_score
4) classification_report

为什么还要写这么一个项目？

例如在预测:

```python
from sklearn.metrics import precision_score
y_true = []
y_pred = []
precision_score(y_true=y_true,y_pred=y_pred)
```

假设`y_true`或者`y_pred`的数据量非常大的时候，那么光是存这些数据就已经要消耗大量内存，更别提计算了。

