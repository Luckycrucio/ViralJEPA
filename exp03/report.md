# exp03 report

## Pretraining

60,000 steps, 2.9 h, last window: jepa 0.7014 (batch-mean baseline 0.8899), mlm 1.312 (acc 0.365), var 0.000, predictor var 0.000, cov 0.0480, std 0.533, token similarity 0.973, mask 0.36.

Frozen linear probe (train fullset_train, score fullset_test, RC-averaged features):

| encoder | AUROC | AP | RankMe |
|---|---|---|---|
| random init | 0.8391 | 0.1297 | 48.7 |
| JEPA (EMA target) | 0.6236 | 0.0351 | 285.2 |

## Fine-tuning (test set, mean ± std over seeds)

| task | init | seeds | AUROC | AP | AUROC RC-TTA | AP RC-TTA | AP 95% CI (RC, seed 0) | F1 RC | precision RC | recall RC |
|---|---|---|---|---|---|---|---|---|---|---|
| fullset | exp01 | 1 | 0.9145 | 0.4639 | 0.9224 | 0.5080 | | | | |
| anell | jepa | 3 | 0.7822 ± 0.0144 | 0.2160 ± 0.0234 | 0.8151 ± 0.0181 | 0.2499 ± 0.0387 | [0.181, 0.221] | 0.1833 ± 0.0330 | 0.5335 ± 0.0720 | 0.1115 ± 0.0234 |
| anell | random | 3 | 0.7881 ± 0.0088 | 0.2303 ± 0.0203 | 0.8164 ± 0.0045 | 0.2711 ± 0.0211 | [0.266, 0.316] | 0.2071 ± 0.0262 | 0.5718 ± 0.0568 | 0.1271 ± 0.0190 |
| fullset | jepa | 3 | 0.9195 ± 0.0008 | 0.5208 ± 0.0107 | 0.9390 ± 0.0018 | 0.5924 ± 0.0071 | [0.560, 0.637] | 0.5754 ± 0.0149 | 0.7356 ± 0.0815 | 0.4767 ± 0.0178 |
| fullset | random | 3 | 0.9185 ± 0.0030 | 0.5571 ± 0.0044 | 0.9331 ± 0.0032 | 0.6067 ± 0.0029 | [0.568, 0.644] | 0.5931 ± 0.0132 | 0.7175 ± 0.0523 | 0.5076 ± 0.0165 |

fullset = fullset_test (551 viral / 26,405). anell = test_data_anell (1,348 viral / 27,644), anelloviruses unseen in pretraining and fine-tuning. Threshold = best F1 on validation.

