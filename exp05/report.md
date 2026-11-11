# exp05 report

## Pretraining

60,000 steps, 2.1 h, last window (batch-mean baselines in brackets): jepa 0.6834 (0.8028), global 0.5542 (0.6319), kmer 0.3325 (0.4110), mlm 1.350 (acc 0.329), var 0.007, cov 2.1050, CLS std 0.301, token similarity 0.536, mask 0.36.

Frozen linear probe (train fullset_train, score fullset_test, RC-averaged features):

| encoder | features | AUROC | AP | RankMe |
|---|---|---|---|---|
| random init | CLS | 0.8404 | 0.1349 | 57.3 |
| random init | mean-pool | 0.8431 | 0.1249 | 47.8 |
| JEPA (EMA target) | CLS | 0.9193 | 0.3865 | 151.7 |
| JEPA (EMA target) | mean-pool | 0.9000 | 0.3076 | 149.6 |

## Fine-tuning (test set, mean ± std over seeds)

| task | init | seeds | AUROC | AP | AUROC RC-TTA | AP RC-TTA | AP 95% CI (RC, seed 0) | F1 RC | precision RC | recall RC |
|---|---|---|---|---|---|---|---|---|---|---|
| fullset | exp01 | 1 | 0.9145 | 0.4639 | 0.9224 | 0.5080 | | | | |
| anell | jepa | 3 | 0.8703 ± 0.0191 | 0.4699 ± 0.0294 | 0.8854 ± 0.0155 | 0.5079 ± 0.0317 | [0.446, 0.501] | 0.3775 ± 0.0442 | 0.7787 ± 0.0122 | 0.2502 ± 0.0380 |
| anell | random | 3 | 0.7908 ± 0.0032 | 0.2307 ± 0.0094 | 0.8216 ± 0.0069 | 0.2684 ± 0.0064 | [0.254, 0.302] | 0.1822 ± 0.0067 | 0.5870 ± 0.0333 | 0.1081 ± 0.0057 |
| fullset | jepa | 3 | 0.9302 ± 0.0157 | 0.6024 ± 0.0041 | 0.9393 ± 0.0113 | 0.6258 ± 0.0076 | [0.585, 0.658] | 0.5827 ± 0.0095 | 0.7043 ± 0.0602 | 0.4997 ± 0.0181 |
| fullset | random | 3 | 0.9181 ± 0.0048 | 0.5732 ± 0.0047 | 0.9314 ± 0.0033 | 0.6167 ± 0.0051 | [0.568, 0.646] | 0.5942 ± 0.0062 | 0.7748 ± 0.0470 | 0.4840 ± 0.0233 |

fullset = fullset_test (551 viral / 26,405). anell = test_data_anell (1,348 viral / 27,644), anelloviruses unseen in pretraining and fine-tuning. Threshold = best F1 on validation.

