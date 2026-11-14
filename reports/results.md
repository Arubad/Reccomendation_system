# Results

## Headline (500m, test split, Listen+ truth and warm users)

| model | ndcg@10 | ndcg@100 | recall@10 | recall@100 | recall_paper@10 | recall_paper@100 | coverage@10 |
|---|---|---|---|---|---|---|---|
| a8_full_all_ids | 0.1009 | 0.1322 | 0.0980 | 0.1909 | 0.0591 | 0.1892 | 0.0236 |
| decaypop | 0.0263 | 0.0365 | 0.0276 | 0.0587 | 0.0150 | 0.0582 | 0.0000 |
| ials | 0.0392 | 0.0677 | 0.0401 | 0.1136 | 0.0181 | 0.1123 | 0.0052 |
| mostpop | 0.0161 | 0.0223 | 0.0153 | 0.0349 | 0.0068 | 0.0345 | 0.0000 |
| sasrec | 0.0798 | 0.1005 | 0.0769 | 0.1426 | 0.0489 | 0.1416 | 0.0488 |
| MostPop (paper) | 0.0173 | 0.0237 | - | - | 0.0060 | 0.0309 | 0.0000 |
| DecayPop (paper) | 0.0267 | 0.0340 | - | - | 0.0127 | 0.0505 | 0.0000 |
| ItemKNN (paper) | 0.0708 | 0.0771 | - | - | 0.0320 | 0.1049 | 0.0257 |
| iALS (paper) | 0.0384 | 0.0621 | - | - | 0.0131 | 0.0811 | 0.0022 |
| BPR (paper) | 0.0400 | 0.0652 | - | - | 0.0137 | 0.0861 | 0.0244 |
| SASRec (paper) | 0.0754 | 0.0884 | - | - | 0.0336 | 0.1240 | 0.0347 |

_Recall definitions._ `recall@k` is the official Yambda definition: hits@k / min(|T|, k) (benchmarks/yambda/evaluation/metrics.py, after the bug-fix commit "Fix yet another bug in metrics (#24)", whose leftover comment suggests the denominator used to be |T|). `recall_paper@k` is hits@k / |T|. The paper's Recall column is much closer to hits/|T| than to min(|T|, k), though not equal to it: our NDCG@10 agrees with the paper within 7%, our min(|T|, k) recall is 2.0-3.1x the paper's at k=10 (1.12-1.46x at k=100), and hits/|T| is 12-38% above it. So paper rows are shown under `recall_paper@k` and their `recall@k` is left as "-"; compare paper Recall against `recall_paper@k` only. Older run records without `recall_paper@k` show "-".

## Headline (50m, test split, Listen+ truth and warm users)

| model | ndcg@10 | ndcg@100 | recall@10 | recall@100 | recall_paper@10 | recall_paper@100 | coverage@10 |
|---|---|---|---|---|---|---|---|
| a0_sasrec | 0.1074 | 0.1299 | 0.1052 | 0.1823 | 0.0664 | 0.1809 | 0.0319 |
| a1_audio | 0.0967 | 0.1161 | 0.0928 | 0.1626 | 0.0580 | 0.1613 | 0.0351 |
| a2_audio_event_organic | 0.1002 | 0.1186 | 0.0968 | 0.1657 | 0.0609 | 0.1644 | 0.0346 |
| a3_full_logq | 0.1151 | 0.1433 | 0.1116 | 0.2016 | 0.0692 | 0.1998 | 0.0233 |
| a4_full_dislike | 0.1163 | 0.1436 | 0.1122 | 0.2001 | 0.0690 | 0.1983 | 0.0244 |
| a7_full_no_content | 0.1207 | 0.1487 | 0.1181 | 0.2085 | 0.0742 | 0.2068 | 0.0224 |
| a8_full_all_ids | 0.1184 | 0.1452 | 0.1170 | 0.2046 | 0.0733 | 0.2029 | 0.0248 |
| decaypop | 0.0256 | 0.0350 | 0.0264 | 0.0565 | - | - | 0.0000 |
| ials | 0.0379 | 0.0696 | 0.0384 | 0.1179 | - | - | 0.0083 |
| mostpop | 0.0177 | 0.0231 | 0.0167 | 0.0359 | - | - | 0.0000 |
| sasrec | 0.0970 | 0.1132 | 0.0939 | 0.1551 | 0.0611 | 0.1541 | 0.0341 |
| sasrec_bce | 0.0771 | 0.0909 | 0.0662 | 0.1211 | 0.0409 | 0.1200 | 0.0094 |
| MostPop (paper) | 0.0186 | 0.0249 | - | - | 0.0064 | 0.0321 | 0.0000 |
| DecayPop (paper) | 0.0260 | 0.0323 | - | - | 0.0122 | 0.0479 | 0.0000 |
| ItemKNN (paper) | 0.0781 | 0.0934 | - | - | 0.0373 | 0.1297 | 0.0180 |
| iALS (paper) | 0.0407 | 0.0642 | - | - | 0.0128 | 0.0808 | 0.0044 |
| BPR (paper) | 0.0389 | 0.0641 | - | - | 0.0139 | 0.0836 | 0.0257 |
| SANSA (paper) | 0.0069 | 0.0095 | - | - | 0.0031 | 0.0137 | 0.0114 |
| SASRec (paper) | 0.0748 | 0.0764 | - | - | 0.0325 | 0.1026 | 0.0130 |

_Recall definitions._ `recall@k` is the official Yambda definition: hits@k / min(|T|, k) (benchmarks/yambda/evaluation/metrics.py, after the bug-fix commit "Fix yet another bug in metrics (#24)", whose leftover comment suggests the denominator used to be |T|). `recall_paper@k` is hits@k / |T|. The paper's Recall column is much closer to hits/|T| than to min(|T|, k), though not equal to it: our NDCG@10 agrees with the paper within 7%, our min(|T|, k) recall is 2.0-3.1x the paper's at k=10 (1.12-1.46x at k=100), and hits/|T| is 12-38% above it. So paper rows are shown under `recall_paper@k` and their `recall@k` is left as "-"; compare paper Recall against `recall_paper@k` only. Older run records without `recall_paper@k` show "-".

## Ablations (500m)

| run | ndcg@10 | d ndcg@10 | recall@100 | d recall@100 |
|---|---|---|---|---|
| a8_full_all_ids | 0.1009 | n/a | 0.1909 | n/a |

## Ablations (50m)

| run | ndcg@10 | d ndcg@10 | recall@100 | d recall@100 |
|---|---|---|---|---|
| a0_sasrec | 0.1074 | - | 0.1823 | - |
| a1_audio | 0.0967 | -0.0107 | 0.1626 | -0.0197 |
| a2_audio_event_organic | 0.1002 | -0.0072 | 0.1657 | -0.0166 |
| a3_full_logq | 0.1151 | +0.0077 | 0.2016 | +0.0193 |
| a4_full_dislike | 0.1163 | +0.0089 | 0.2001 | +0.0178 |
| a7_full_no_content | 0.1207 | +0.0132 | 0.2085 | +0.0262 |
| a8_full_all_ids | 0.1184 | +0.0110 | 0.2046 | +0.0223 |

## Slices

_Slices._ `organic` / `served`: truth items with an organic / recommender-served test event. `idless_items`: truth items without an ID in the data vocab (fewer than `data.min_count` train positives) or out of vocabulary; the same items for every model, whatever its own ID table (`model.id_min_count`); records before F5 named it `cold_items`. `new_items`: truth items with no train-period event of any kind (first seen on the test day), a subset of `idless_items`; only content features can score them. `warm_items`: truth items with a data-vocab ID. `users_low/mid/high`: terciles of the users' Listen+ train events.

### a8_full_all_ids-test (500m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 45921 | 0.1009 | 0.0980 | 0.1322 | 0.1909 | 0.0236 |
| organic | 31907 | 0.1239 | 0.1231 | 0.1632 | 0.2442 | 0.0204 |
| served | 28491 | 0.0336 | 0.0359 | 0.0626 | 0.1112 | 0.0169 |
| idless_items | 4852 | 0.0139 | 0.0174 | 0.0182 | 0.0333 | 0.0091 |
| new_items | 2006 | 0.0003 | 0.0005 | 0.0023 | 0.0089 | 0.0046 |
| warm_items | 45765 | 0.1007 | 0.0980 | 0.1325 | 0.1922 | 0.0234 |
| users_low | 15307 | 0.1089 | 0.1106 | 0.1554 | 0.2426 | 0.0127 |
| users_mid | 15307 | 0.0952 | 0.0913 | 0.1280 | 0.1858 | 0.0130 |
| users_high | 15307 | 0.0987 | 0.0922 | 0.1131 | 0.1443 | 0.0133 |
| cold users (n=8) | - | 0.1250 | 0.1250 | 0.1468 | 0.1789 | 0.0000 |

### decaypop-test (500m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 45921 | 0.0263 | 0.0276 | 0.0365 | 0.0587 | 0.0000 |
| organic | 31907 | 0.0278 | 0.0329 | 0.0433 | 0.0791 | 0.0000 |
| served | 28491 | 0.0157 | 0.0161 | 0.0242 | 0.0414 | 0.0000 |
| idless_items | 4852 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| new_items | 2006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| warm_items | 45765 | 0.0264 | 0.0277 | 0.0367 | 0.0592 | 0.0000 |
| users_low | 15307 | 0.0252 | 0.0285 | 0.0408 | 0.0744 | 0.0000 |
| users_mid | 15307 | 0.0242 | 0.0255 | 0.0350 | 0.0564 | 0.0000 |
| users_high | 15307 | 0.0294 | 0.0288 | 0.0336 | 0.0452 | 0.0000 |
| cold users (n=8) | - | 0.1250 | 0.1250 | 0.1418 | 0.1721 | 0.0000 |

### ials-test (500m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 45921 | 0.0393 | 0.0401 | 0.0677 | 0.1135 | 0.0052 |
| organic | 31907 | 0.0336 | 0.0355 | 0.0652 | 0.1222 | 0.0047 |
| served | 28491 | 0.0280 | 0.0293 | 0.0519 | 0.0935 | 0.0048 |
| idless_items | 4852 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0035 |
| new_items | 2006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0024 |
| warm_items | 45765 | 0.0395 | 0.0403 | 0.0682 | 0.1145 | 0.0052 |
| users_low | 15307 | 0.0489 | 0.0504 | 0.0868 | 0.1557 | 0.0028 |
| users_mid | 15307 | 0.0349 | 0.0357 | 0.0640 | 0.1086 | 0.0035 |
| users_high | 15307 | 0.0341 | 0.0343 | 0.0525 | 0.0763 | 0.0048 |
| cold users (n=8) | - | 0.0984 | 0.0875 | 0.0959 | 0.1134 | 0.0000 |

### mostpop-test (500m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 45921 | 0.0161 | 0.0153 | 0.0223 | 0.0349 | 0.0000 |
| organic | 31907 | 0.0164 | 0.0166 | 0.0240 | 0.0415 | 0.0000 |
| served | 28491 | 0.0092 | 0.0091 | 0.0157 | 0.0277 | 0.0000 |
| idless_items | 4852 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| new_items | 2006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| warm_items | 45765 | 0.0162 | 0.0154 | 0.0224 | 0.0352 | 0.0000 |
| users_low | 15307 | 0.0172 | 0.0169 | 0.0267 | 0.0461 | 0.0000 |
| users_mid | 15307 | 0.0149 | 0.0141 | 0.0214 | 0.0337 | 0.0000 |
| users_high | 15307 | 0.0163 | 0.0149 | 0.0187 | 0.0248 | 0.0000 |
| cold users (n=8) | - | 0.0984 | 0.0875 | 0.0959 | 0.1134 | 0.0000 |

### sasrec-test (500m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 45921 | 0.0798 | 0.0769 | 0.1005 | 0.1426 | 0.0488 |
| organic | 31907 | 0.1046 | 0.1028 | 0.1317 | 0.1927 | 0.0406 |
| served | 28491 | 0.0189 | 0.0207 | 0.0379 | 0.0685 | 0.0355 |
| idless_items | 4852 | 0.0273 | 0.0354 | 0.0304 | 0.0498 | 0.0132 |
| new_items | 2006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0060 |
| warm_items | 45765 | 0.0791 | 0.0765 | 0.1004 | 0.1434 | 0.0486 |
| users_low | 15307 | 0.0877 | 0.0887 | 0.1197 | 0.1834 | 0.0244 |
| users_mid | 15307 | 0.0754 | 0.0719 | 0.0976 | 0.1387 | 0.0258 |
| users_high | 15307 | 0.0764 | 0.0700 | 0.0843 | 0.1058 | 0.0258 |
| cold users (n=8) | - | 0.0984 | 0.0875 | 0.0959 | 0.1134 | 0.0000 |

### a0_sasrec-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.1074 | 0.1052 | 0.1299 | 0.1823 | 0.0319 |
| organic | 3236 | 0.1436 | 0.1441 | 0.1773 | 0.2570 | 0.0248 |
| served | 2858 | 0.0191 | 0.0211 | 0.0381 | 0.0704 | 0.0218 |
| idless_items | 1270 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0122 |
| new_items | 475 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0050 |
| warm_items | 4566 | 0.1087 | 0.1067 | 0.1327 | 0.1877 | 0.0317 |
| users_low | 1533 | 0.1196 | 0.1232 | 0.1580 | 0.2419 | 0.0131 |
| users_mid | 1532 | 0.0996 | 0.0958 | 0.1230 | 0.1690 | 0.0137 |
| users_high | 1532 | 0.1030 | 0.0966 | 0.1086 | 0.1361 | 0.0138 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### a1_audio-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.0967 | 0.0928 | 0.1161 | 0.1626 | 0.0351 |
| organic | 3236 | 0.1272 | 0.1250 | 0.1570 | 0.2288 | 0.0269 |
| served | 2858 | 0.0189 | 0.0207 | 0.0353 | 0.0635 | 0.0237 |
| idless_items | 1270 | 0.0082 | 0.0157 | 0.0227 | 0.0749 | 0.0129 |
| new_items | 475 | 0.0037 | 0.0058 | 0.0161 | 0.0577 | 0.0051 |
| warm_items | 4566 | 0.0966 | 0.0927 | 0.1163 | 0.1633 | 0.0349 |
| users_low | 1533 | 0.1055 | 0.1066 | 0.1382 | 0.2100 | 0.0140 |
| users_mid | 1532 | 0.0926 | 0.0878 | 0.1123 | 0.1532 | 0.0144 |
| users_high | 1532 | 0.0922 | 0.0840 | 0.0977 | 0.1246 | 0.0145 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### a2_audio_event_organic-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.1002 | 0.0968 | 0.1186 | 0.1657 | 0.0346 |
| organic | 3236 | 0.1314 | 0.1307 | 0.1605 | 0.2341 | 0.0265 |
| served | 2858 | 0.0205 | 0.0218 | 0.0366 | 0.0641 | 0.0233 |
| idless_items | 1270 | 0.0098 | 0.0179 | 0.0221 | 0.0666 | 0.0128 |
| new_items | 475 | 0.0047 | 0.0070 | 0.0152 | 0.0480 | 0.0051 |
| warm_items | 4566 | 0.0999 | 0.0965 | 0.1190 | 0.1669 | 0.0344 |
| users_low | 1533 | 0.1091 | 0.1111 | 0.1415 | 0.2159 | 0.0138 |
| users_mid | 1532 | 0.0950 | 0.0902 | 0.1141 | 0.1552 | 0.0142 |
| users_high | 1532 | 0.0963 | 0.0890 | 0.1002 | 0.1260 | 0.0144 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### a3_full_logq-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.1152 | 0.1116 | 0.1433 | 0.2016 | 0.0233 |
| organic | 3236 | 0.1479 | 0.1466 | 0.1850 | 0.2693 | 0.0184 |
| served | 2858 | 0.0273 | 0.0303 | 0.0542 | 0.0978 | 0.0165 |
| idless_items | 1270 | 0.0061 | 0.0103 | 0.0131 | 0.0376 | 0.0111 |
| new_items | 475 | 0.0034 | 0.0060 | 0.0076 | 0.0213 | 0.0048 |
| warm_items | 4566 | 0.1152 | 0.1117 | 0.1445 | 0.2044 | 0.0231 |
| users_low | 1533 | 0.1282 | 0.1290 | 0.1721 | 0.2621 | 0.0101 |
| users_mid | 1532 | 0.1081 | 0.1027 | 0.1382 | 0.1926 | 0.0108 |
| users_high | 1532 | 0.1091 | 0.1031 | 0.1194 | 0.1499 | 0.0114 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### a4_full_dislike-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.1163 | 0.1122 | 0.1436 | 0.2001 | 0.0244 |
| organic | 3236 | 0.1479 | 0.1452 | 0.1858 | 0.2696 | 0.0192 |
| served | 2858 | 0.0281 | 0.0302 | 0.0537 | 0.0949 | 0.0172 |
| idless_items | 1270 | 0.0066 | 0.0110 | 0.0145 | 0.0417 | 0.0113 |
| new_items | 475 | 0.0039 | 0.0071 | 0.0097 | 0.0312 | 0.0049 |
| warm_items | 4566 | 0.1163 | 0.1123 | 0.1447 | 0.2027 | 0.0243 |
| users_low | 1533 | 0.1271 | 0.1290 | 0.1709 | 0.2578 | 0.0104 |
| users_mid | 1532 | 0.1121 | 0.1060 | 0.1405 | 0.1920 | 0.0112 |
| users_high | 1532 | 0.1096 | 0.1015 | 0.1194 | 0.1505 | 0.0117 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### a7_full_no_content-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.1207 | 0.1181 | 0.1487 | 0.2085 | 0.0224 |
| organic | 3236 | 0.1578 | 0.1580 | 0.1950 | 0.2833 | 0.0177 |
| served | 2858 | 0.0252 | 0.0276 | 0.0536 | 0.0994 | 0.0162 |
| idless_items | 1270 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0107 |
| new_items | 475 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0047 |
| warm_items | 4566 | 0.1221 | 0.1199 | 0.1519 | 0.2143 | 0.0222 |
| users_low | 1533 | 0.1352 | 0.1388 | 0.1783 | 0.2697 | 0.0099 |
| users_mid | 1532 | 0.1113 | 0.1066 | 0.1434 | 0.2003 | 0.0107 |
| users_high | 1532 | 0.1155 | 0.1088 | 0.1245 | 0.1554 | 0.0112 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### a8_full_all_ids-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.1184 | 0.1170 | 0.1452 | 0.2046 | 0.0248 |
| organic | 3236 | 0.1541 | 0.1550 | 0.1903 | 0.2758 | 0.0195 |
| served | 2858 | 0.0258 | 0.0287 | 0.0528 | 0.0990 | 0.0175 |
| idless_items | 1270 | 0.0146 | 0.0214 | 0.0215 | 0.0473 | 0.0114 |
| new_items | 475 | 0.0012 | 0.0020 | 0.0057 | 0.0188 | 0.0048 |
| warm_items | 4566 | 0.1178 | 0.1167 | 0.1459 | 0.2071 | 0.0247 |
| users_low | 1533 | 0.1344 | 0.1386 | 0.1764 | 0.2667 | 0.0105 |
| users_mid | 1532 | 0.1081 | 0.1067 | 0.1396 | 0.1969 | 0.0113 |
| users_high | 1532 | 0.1128 | 0.1058 | 0.1196 | 0.1500 | 0.0118 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### decaypop-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.0256 | 0.0264 | 0.0350 | 0.0565 | 0.0000 |
| organic | 3236 | 0.0273 | 0.0307 | 0.0416 | 0.0758 | 0.0000 |
| served | 2858 | 0.0145 | 0.0155 | 0.0232 | 0.0407 | 0.0000 |
| idless_items | 1270 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| new_items | 475 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| warm_items | 4566 | 0.0259 | 0.0267 | 0.0355 | 0.0575 | 0.0000 |
| users_low | 1533 | 0.0235 | 0.0265 | 0.0389 | 0.0709 | 0.0000 |
| users_mid | 1532 | 0.0260 | 0.0263 | 0.0352 | 0.0562 | 0.0000 |
| users_high | 1532 | 0.0275 | 0.0264 | 0.0309 | 0.0424 | 0.0000 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### ials-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.0379 | 0.0384 | 0.0696 | 0.1179 | 0.0083 |
| organic | 3236 | 0.0331 | 0.0351 | 0.0691 | 0.1303 | 0.0069 |
| served | 2858 | 0.0256 | 0.0263 | 0.0491 | 0.0876 | 0.0069 |
| idless_items | 1270 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0056 |
| new_items | 475 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0031 |
| warm_items | 4566 | 0.0383 | 0.0390 | 0.0710 | 0.1210 | 0.0082 |
| users_low | 1533 | 0.0518 | 0.0534 | 0.0895 | 0.1597 | 0.0033 |
| users_mid | 1532 | 0.0368 | 0.0367 | 0.0698 | 0.1178 | 0.0046 |
| users_high | 1532 | 0.0251 | 0.0251 | 0.0495 | 0.0763 | 0.0067 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### mostpop-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.0177 | 0.0167 | 0.0231 | 0.0359 | 0.0000 |
| organic | 3236 | 0.0175 | 0.0179 | 0.0247 | 0.0429 | 0.0000 |
| served | 2858 | 0.0104 | 0.0103 | 0.0164 | 0.0280 | 0.0000 |
| idless_items | 1270 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| new_items | 475 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| warm_items | 4566 | 0.0178 | 0.0168 | 0.0234 | 0.0366 | 0.0000 |
| users_low | 1533 | 0.0187 | 0.0183 | 0.0285 | 0.0487 | 0.0000 |
| users_mid | 1532 | 0.0161 | 0.0149 | 0.0213 | 0.0327 | 0.0000 |
| users_high | 1532 | 0.0183 | 0.0169 | 0.0195 | 0.0262 | 0.0000 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### sasrec-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.0970 | 0.0940 | 0.1132 | 0.1551 | 0.0341 |
| organic | 3236 | 0.1317 | 0.1302 | 0.1572 | 0.2228 | 0.0263 |
| served | 2858 | 0.0142 | 0.0154 | 0.0292 | 0.0558 | 0.0233 |
| idless_items | 1270 | 0.0184 | 0.0244 | 0.0238 | 0.0479 | 0.0128 |
| new_items | 475 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0051 |
| warm_items | 4566 | 0.0960 | 0.0936 | 0.1136 | 0.1575 | 0.0339 |
| users_low | 1533 | 0.1095 | 0.1114 | 0.1392 | 0.2067 | 0.0137 |
| users_mid | 1532 | 0.0908 | 0.0868 | 0.1098 | 0.1478 | 0.0142 |
| users_high | 1532 | 0.0908 | 0.0838 | 0.0907 | 0.1108 | 0.0144 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### sasrec_bce-test (50m)

| slice | users | ndcg@10 | recall@10 | ndcg@100 | recall@100 | coverage@10 (within slice) |
|---|---|---|---|---|---|---|
| all | 4597 | 0.0771 | 0.0662 | 0.0909 | 0.1211 | 0.0094 |
| organic | 3236 | 0.0958 | 0.0813 | 0.1132 | 0.1555 | 0.0076 |
| served | 2858 | 0.0222 | 0.0241 | 0.0403 | 0.0718 | 0.0071 |
| idless_items | 1270 | 0.0060 | 0.0064 | 0.0072 | 0.0111 | 0.0047 |
| new_items | 475 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0024 |
| warm_items | 4566 | 0.0773 | 0.0667 | 0.0920 | 0.1234 | 0.0094 |
| users_low | 1533 | 0.0854 | 0.0773 | 0.1095 | 0.1578 | 0.0046 |
| users_mid | 1532 | 0.0723 | 0.0612 | 0.0877 | 0.1162 | 0.0049 |
| users_high | 1532 | 0.0734 | 0.0603 | 0.0754 | 0.0893 | 0.0051 |
| cold users (n=1) | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Fairness and confidence intervals

### 500m

| run | artist_gini@10 | artist_coverage@10 | long_tail_share@10 | artist_gini@100 | artist_coverage@100 | long_tail_share@100 | ndcg@10 [95% CI] | recall@100 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| a8_full_all_ids-test | 0.9927 | 0.0394 | 0.0170 | 0.9914 | 0.0849 | 0.0172 | 0.1009 [0.0992, 0.1026] | 0.1909 [0.1886, 0.1933] |
| decaypop-test | 1.0000 | 0.0000 | 0.0000 | 0.9999 | 0.0002 | 0.0000 | 0.0263 [0.0255, 0.0271] | 0.0587 [0.0574, 0.0599] |
| ials-test | 0.9976 | 0.0127 | 0.0000 | 0.9970 | 0.0211 | 0.0000 | 0.0393 [0.0384, 0.0402] | 0.1135 [0.1119, 0.1150] |
| mostpop-test | 1.0000 | 0.0000 | 0.0000 | 0.9999 | 0.0002 | 0.0000 | 0.0161 [0.0156, 0.0167] | 0.0349 [0.0341, 0.0357] |
| sasrec-test | 0.9845 | 0.0686 | 0.0416 | 0.9836 | 0.1131 | 0.0250 | 0.0798 [0.0784, 0.0814] | 0.1426 [0.1407, 0.1447] |

### 50m

| run | artist_gini@10 | artist_coverage@10 | long_tail_share@10 | artist_gini@100 | artist_coverage@100 | long_tail_share@100 | ndcg@10 [95% CI] | recall@100 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| a0_sasrec-test | 0.9723 | 0.0700 | 0.0622 | 0.9667 | 0.1336 | 0.0432 | 0.1074 [0.1023, 0.1132] | 0.1823 [0.1751, 0.1900] |
| a1_audio-test | 0.9723 | 0.0688 | 0.1480 | 0.9650 | 0.1486 | 0.2101 | 0.0967 [0.0914, 0.1023] | 0.1626 [0.1563, 0.1695] |
| a2_audio_event_organic-test | 0.9724 | 0.0685 | 0.1436 | 0.9649 | 0.1485 | 0.2022 | 0.1002 [0.0950, 0.1054] | 0.1657 [0.1587, 0.1725] |
| a3_full_logq-test | 0.9820 | 0.0521 | 0.0490 | 0.9772 | 0.1165 | 0.0649 | 0.1152 [0.1095, 0.1210] | 0.2016 [0.1947, 0.2094] |
| a4_full_dislike-test | 0.9811 | 0.0544 | 0.0572 | 0.9767 | 0.1186 | 0.0751 | 0.1163 [0.1106, 0.1225] | 0.2001 [0.1932, 0.2077] |
| a7_full_no_content-test | 0.9818 | 0.0534 | 0.0289 | 0.9785 | 0.1068 | 0.0170 | 0.1207 [0.1148, 0.1266] | 0.2085 [0.2013, 0.2162] |
| a8_full_all_ids-test | 0.9808 | 0.0555 | 0.0608 | 0.9760 | 0.1259 | 0.0578 | 0.1184 [0.1129, 0.1246] | 0.2046 [0.1975, 0.2120] |
| decaypop-test | 1.0000 | 0.0001 | 0.0000 | 0.9996 | 0.0005 | 0.0000 | 0.0256 [0.0235, 0.0281] | 0.0565 [0.0530, 0.0606] |
| ials-test | 0.9938 | 0.0239 | 0.0000 | 0.9910 | 0.0512 | 0.0000 | 0.0379 [0.0350, 0.0408] | 0.1179 [0.1131, 0.1228] |
| mostpop-test | 0.9999 | 0.0001 | 0.0000 | 0.9996 | 0.0005 | 0.0000 | 0.0177 [0.0157, 0.0198] | 0.0359 [0.0333, 0.0388] |
| sasrec-test | 0.9701 | 0.0742 | 0.1029 | 0.9625 | 0.1556 | 0.0815 | 0.0970 [0.0915, 0.1025] | 0.1551 [0.1484, 0.1615] |
| sasrec_bce-test | 0.9943 | 0.0270 | 0.0091 | 0.9934 | 0.0471 | 0.0019 | 0.0771 [0.0725, 0.0812] | 0.1211 [0.1157, 0.1264] |

## Paired difference: HookSeq vs SASRec

### 500m

_HookSeq = `a8_full_all_ids` (the only HookSeq-model run at this size)._

| comparison | metric | mean diff | 95% CI |
|---|---|---|---|
| HookSeq (a8_full_all_ids) vs sasrec (strong SASRec, Listen+ input) | ndcg@10 (a8_full_all_ids - sasrec) | +0.0211 | [+0.0198, +0.0223] |
| HookSeq (a8_full_all_ids) vs sasrec (strong SASRec, Listen+ input) | recall@100 (a8_full_all_ids - sasrec) | +0.0483 | [+0.0467, +0.0499] |

### 50m

_HookSeq = `a4_full_dislike` (no run named hookseq; a4_full_dislike's config equals hookseq.yaml)._

| comparison | metric | mean diff | 95% CI |
|---|---|---|---|
| HookSeq (a4_full_dislike) vs sasrec (strong SASRec, Listen+ input) | ndcg@10 (a4_full_dislike - sasrec) | +0.0192 | [+0.0152, +0.0232] |
| HookSeq (a4_full_dislike) vs sasrec (strong SASRec, Listen+ input) | recall@100 (a4_full_dislike - sasrec) | +0.0450 | [+0.0398, +0.0501] |
| HookSeq (a4_full_dislike) vs a0_sasrec (a0 control, same code path) | ndcg@10 (a4_full_dislike - a0_sasrec) | +0.0089 | [+0.0052, +0.0127] |
| HookSeq (a4_full_dislike) vs a0_sasrec (a0 control, same code path) | recall@100 (a4_full_dislike - a0_sasrec) | +0.0178 | [+0.0129, +0.0228] |
| a7_full_no_content vs sasrec (strong SASRec, Listen+ input) | ndcg@10 (a7_full_no_content - sasrec) | +0.0236 | [+0.0196, +0.0277] |
| a7_full_no_content vs sasrec (strong SASRec, Listen+ input) | recall@100 (a7_full_no_content - sasrec) | +0.0533 | [+0.0483, +0.0583] |
| a7_full_no_content vs a0_sasrec (a0 control, same code path) | ndcg@10 (a7_full_no_content - a0_sasrec) | +0.0133 | [+0.0099, +0.0167] |
| a7_full_no_content vs a0_sasrec (a0 control, same code path) | recall@100 (a7_full_no_content - a0_sasrec) | +0.0261 | [+0.0216, +0.0305] |
| a8_full_all_ids vs sasrec (strong SASRec, Listen+ input) | ndcg@10 (a8_full_all_ids - sasrec) | +0.0214 | [+0.0175, +0.0252] |
| a8_full_all_ids vs sasrec (strong SASRec, Listen+ input) | recall@100 (a8_full_all_ids - sasrec) | +0.0494 | [+0.0446, +0.0546] |
| a8_full_all_ids vs a0_sasrec (a0 control, same code path) | ndcg@10 (a8_full_all_ids - a0_sasrec) | +0.0110 | [+0.0075, +0.0144] |
| a8_full_all_ids vs a0_sasrec (a0 control, same code path) | recall@100 (a8_full_all_ids - a0_sasrec) | +0.0222 | [+0.0173, +0.0269] |
