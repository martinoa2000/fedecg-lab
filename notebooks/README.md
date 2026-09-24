Numbered notebooks. Each explains the concepts in markdown before the code.
Reusable logic lives in src/fedecg/ and is imported here.

01_exploration           label balance, co-occurrence, folds, preprocessing (phase 2)
02_centralized_baseline  recipe selection, test results vs. published (phase 3)
03_federated_iid         FedAvg across 5, 10 and 20 random hospitals (phase 4)
04_federated_non_iid     site, device and label-skew hospitals; FedAvg vs. FedProx (phase 5)
05_differential_privacy  DP-SGD at epsilon 1, 3, 8, centralized and per hospital (phase 6)
06_explainability        IG and Grad-CAM, beat-segment enrichment, randomization check (phase 7)

Notebooks 02-06 read results/tables/ and train nothing; run the scripts first
(or scripts/reproduce.sh).
