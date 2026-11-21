# Final local starter-kit simulator results — NOT a real competition score

2016 steps (seven days) from reset; 30 shipped scenarios; seed 0; seven workers.
Scores below are negative mean per-step rewards: **lower is better**.

All 30 were held out of the training objective. An asterisk marks the five used for ES parameter promotion; the other 25 are independent of selection.
This is not the official 8640-step horizon, a Kaggle server run, or a competition score.

| Scenario | Promoted | Original defaults | pid_water |
|---|---:|---:|---:|
| building_train_1/atlanta_2 * | 0.00430220 | 0.00970632 | 0.01949447 |
| building_train_1/denver_7 | 0.00530199 | 0.01032976 | 0.02725076 |
| building_train_1/fairbanks_1 | 0.07767512 | 0.26624341 | 0.37866590 |
| building_train_1/phoenix_5 | 0.02176969 | 0.04472744 | 0.12065732 |
| building_train_1/singapore_8 | 0.01187496 | 0.02235297 | 0.03637971 |
| building_train_2/atlanta_3 | 0.00449017 | 0.00897613 | 0.01993217 |
| building_train_2/denver_8 * | 0.01186206 | 0.01775507 | 0.04758492 |
| building_train_2/fairbanks_2 | 0.02612554 | 0.04056133 | 0.13606384 |
| building_train_2/phoenix_6 | 0.01460896 | 0.03560207 | 0.09258454 |
| building_train_2/singapore_1 | 0.00765701 | 0.01630890 | 0.02317672 |
| building_train_4/atlanta_5 | 0.00969233 | 0.01867123 | 0.02791674 |
| building_train_4/denver_2 | 0.00883124 | 0.01561196 | 0.04884704 |
| building_train_4/fairbanks_4 * | 0.00436747 | 0.00902856 | 0.03180739 |
| building_train_4/phoenix_8 | 0.00389263 | 0.00820121 | 0.01634822 |
| building_train_4/singapore_3 | 0.01154427 | 0.02284425 | 0.03864945 |
| building_train_5/atlanta_6 | 0.00864850 | 0.01669311 | 0.04785949 |
| building_train_5/denver_3 | 0.00972959 | 0.01472473 | 0.04314289 |
| building_train_5/fairbanks_5 | 0.00392319 | 0.00679616 | 0.01380874 |
| building_train_5/phoenix_1 * | 0.00645365 | 0.01036047 | 0.03214760 |
| building_train_5/singapore_4 | 0.01002591 | 0.01996262 | 0.07886453 |
| building_train_7/atlanta_8 | 0.01190876 | 0.01761591 | 0.05063869 |
| building_train_7/denver_5 | 0.00540455 | 0.00996092 | 0.01626586 |
| building_train_7/fairbanks_7 | 0.01162113 | 0.01719404 | 0.05691520 |
| building_train_7/phoenix_3 | 0.00859355 | 0.01692896 | 0.02581429 |
| building_train_7/singapore_6 * | 0.01132165 | 0.02385851 | 0.03050755 |
| building_train_8/atlanta_1 | 0.00729500 | 0.01247293 | 0.02292625 |
| building_train_8/denver_6 | 0.00768937 | 0.01375358 | 0.02236401 |
| building_train_8/fairbanks_8 | 0.03010585 | 0.04924715 | 0.08021155 |
| building_train_8/phoenix_4 | 0.01694858 | 0.03732289 | 0.14178552 |
| building_train_8/singapore_7 | 0.01023556 | 0.02062981 | 0.07741955 |
| **Mean (30)** | **0.01279668** | **0.02781475** | **0.06020103** |
| promotion_subset (5) | 0.00766140 | 0.01414179 | 0.03230839 |
| independent_subset (25) | 0.01382374 | 0.03054934 | 0.06577956 |

Promoted mean loss reduction: 53.99% versus original defaults; 78.74% versus pid_water.
Per-scenario wins: 30/30 versus defaults; 30/30 versus pid_water.

ES: 54 generations, 3.5-hour budget; winner generation 50, fixed-subset 864-step validation 0.0073016549839074411.

Raw rewards, component totals and simulator timing are in final_local_promoted.json, final_local_defaults.json and final_local_pid_water.json. Parameter snapshots and full-precision scores are in final_local_summary.json.
