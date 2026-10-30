# exp02 report

## Pretraining

60,000 steps, 3.1 h, last window: jepa 0.0252, mlm 1.379 (acc 0.284), var 0.007, cov 4.0970, mask 0.36.

Frozen linear probe (train fullset_train, score fullset_test, RC-averaged features):

| encoder | AUROC | AP | RankMe |
|---|---|---|---|
| random init | 0.8333 | 0.1112 | 33.2 |
| JEPA (EMA target) | 0.6092 | 0.0313 | 20.6 |

## Fine-tuning (test set, mean ± std over seeds)

| task | init | seeds | AUROC | AP | AUROC RC-TTA | AP RC-TTA | AP 95% CI (RC, seed 0) | F1 RC | precision RC | recall RC |
|---|---|---|---|---|---|---|---|---|---|---|
| fullset | exp01 | 1 | 0.9145 | 0.4639 | 0.9224 | 0.5080 | | | | |
| anell | jepa | 3 | 0.7281 ± 0.0050 | 0.0966 ± 0.0023 | 0.7399 ± 0.0058 | 0.0979 ± 0.0019 | [0.091, 0.105] | 0.0845 ± 0.0173 | 0.0996 ± 0.0063 | 0.0762 ± 0.0249 |
| anell | random | 3 | 0.7724 ± 0.0183 | 0.2278 ± 0.0167 | 0.8028 ± 0.0123 | 0.2550 ± 0.0204 | [0.206, 0.253] | 0.1980 ± 0.0476 | 0.5096 ± 0.0278 | 0.1259 ± 0.0398 |
| fullset | jepa | 3 | 0.8748 ± 0.0048 | 0.1756 ± 0.0241 | 0.8861 ± 0.0038 | 0.2000 ± 0.0315 | [0.169, 0.226] | 0.2839 ± 0.0270 | 0.2304 ± 0.0252 | 0.3702 ± 0.0267 |
| fullset | random | 3 | 0.9241 ± 0.0040 | 0.5756 ± 0.0043 | 0.9341 ± 0.0019 | 0.6093 ± 0.0010 | [0.568, 0.644] | 0.5804 ± 0.0152 | 0.7909 ± 0.0271 | 0.4586 ± 0.0126 |

fullset = fullset_test (551 viral / 26,405). anell = test_data_anell (1,348 viral / 27,644), anelloviruses unseen in pretraining and fine-tuning. Threshold = best F1 on validation.

