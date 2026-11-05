# exp04 report

## Pretraining

60,000 steps, 2.1 h, last window (batch-mean baselines in brackets): jepa 0.7173 (0.8457), global 0.5455 (0.6272), kmer 0.3352 (0.4133), mlm 1.349 (acc 0.330), var 0.007, cov 2.1036, CLS std 0.317, token similarity 0.563, mask 0.36.

Frozen linear probe (train fullset_train, score fullset_test, RC-averaged features):

| encoder | features | AUROC | AP | RankMe |
|---|---|---|---|---|
| random init | CLS | 0.8404 | 0.1349 | 57.3 |
| random init | mean-pool | 0.8431 | 0.1249 | 47.8 |
| JEPA (EMA target) | CLS | 0.9116 | 0.3746 | 158.8 |
| JEPA (EMA target) | mean-pool | 0.8848 | 0.2904 | 160.6 |

## Fine-tuning (test set, mean ± std over seeds)

| task | init | seeds | AUROC | AP | AUROC RC-TTA | AP RC-TTA | AP 95% CI (RC, seed 0) | F1 RC | precision RC | recall RC |
|---|---|---|---|---|---|---|---|---|---|---|
| fullset | exp01 | 1 | 0.9145 | 0.4639 | 0.9224 | 0.5080 | | | | |
| anell | jepa | 3 | 0.8748 ± 0.0237 | 0.4551 ± 0.0231 | 0.8953 ± 0.0194 | 0.4908 ± 0.0299 | [0.507, 0.562] | 0.3797 ± 0.0482 | 0.7547 ± 0.0298 | 0.2567 ± 0.0466 |
| anell | random | 3 | 0.7908 ± 0.0032 | 0.2307 ± 0.0094 | 0.8216 ± 0.0069 | 0.2684 ± 0.0064 | [0.254, 0.302] | 0.1822 ± 0.0067 | 0.5870 ± 0.0333 | 0.1081 ± 0.0057 |
| fullset | jepa | 3 | 0.9318 ± 0.0052 | 0.5978 ± 0.0066 | 0.9422 ± 0.0047 | 0.6254 ± 0.0106 | [0.592, 0.666] | 0.6013 ± 0.0096 | 0.7403 ± 0.0235 | 0.5070 ± 0.0179 |
| fullset | random | 3 | 0.9181 ± 0.0048 | 0.5732 ± 0.0047 | 0.9314 ± 0.0033 | 0.6167 ± 0.0051 | [0.568, 0.646] | 0.5942 ± 0.0062 | 0.7748 ± 0.0470 | 0.4840 ± 0.0233 |

fullset = fullset_test (551 viral / 26,405). anell = test_data_anell (1,348 viral / 27,644), anelloviruses unseen in pretraining and fine-tuning. Threshold = best F1 on validation.

