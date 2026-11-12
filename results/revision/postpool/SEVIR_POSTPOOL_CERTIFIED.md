# SEVIR, 450 contrasts under the post-pooling control: paired event-bootstrap intervals

Evaluation events 936; control fixed on the calibration events; B = 2000 event resamples (seed 0), the same resample for every arm; percentile 95% intervals.

| subset | n | r(gap, distortion) | ρ | flips | flips with both margins separated |
|---|---|---|---|---|---|
| all | 450 | 0.896 | 0.860 | 64 | 46 |

A margin is separated when its 95% interval excludes zero.

## Flips

| b − a | pooling | τ | B a / b | raw [95% CI] | calibrated [95% CI] | both separated |
|---|---|---|---|---|---|---|
| earthformer − persistence (published) | max16 | 160 | 0.99 / 0.39 | -0.0014 [-0.0135, +0.0112] | +0.1215 [+0.1120, +0.1310] |  |
| cascast_det − persistence | max16 | 160 | 0.99 / 0.34 | -0.0330 [-0.0455, -0.0201] | +0.1229 [+0.1137, +0.1321] | yes |
| cascast_det − persistence | max16 | 181 | 1.00 / 0.28 | -0.0259 [-0.0401, -0.0114] | +0.1223 [+0.1108, +0.1338] | yes |
| cascast_det − persistence | max16 | 219 | 1.02 / 0.17 | -0.0458 [-0.0641, -0.0258] | +0.1319 [+0.1173, +0.1464] | yes |
| earthformer − pysteps | max16 | 74 | 0.92 / 0.75 | -0.0033 [-0.0080, +0.0020] | +0.0423 [+0.0390, +0.0457] |  |
| earthformer − pysteps | max16 | 133 | 0.91 / 0.53 | -0.0648 [-0.0727, -0.0568] | +0.0406 [+0.0366, +0.0448] | yes |
| earthformer − pysteps | max16 | 160 | 0.91 / 0.39 | -0.0918 [-0.1009, -0.0829] | +0.0269 [+0.0213, +0.0327] | yes |
| earthformer − pysteps | max16 | 181 | 0.92 / 0.34 | -0.0811 [-0.0904, -0.0713] | +0.0275 [+0.0204, +0.0347] | yes |
| earthformer − pysteps | max16 | 219 | 0.91 / 0.28 | -0.0620 [-0.0757, -0.0482] | +0.0429 [+0.0338, +0.0516] | yes |
| cascast_det − pysteps | max4 | 219 | 0.90 / 0.21 | -0.0087 [-0.0218, +0.0047] | +0.0903 [+0.0809, +0.0998] |  |
| cascast_det − pysteps | max16 | 74 | 0.92 / 0.72 | -0.0183 [-0.0237, -0.0122] | +0.0429 [+0.0394, +0.0463] | yes |
| cascast_det − pysteps | max16 | 133 | 0.91 / 0.49 | -0.0885 [-0.0971, -0.0798] | +0.0419 [+0.0379, +0.0460] | yes |
| cascast_det − pysteps | max16 | 160 | 0.91 / 0.34 | -0.1233 [-0.1331, -0.1137] | +0.0284 [+0.0228, +0.0340] | yes |
| cascast_det − pysteps | max16 | 181 | 0.92 / 0.28 | -0.1134 [-0.1231, -0.1035] | +0.0322 [+0.0249, +0.0394] | yes |
| cascast_det − pysteps | max16 | 219 | 0.91 / 0.17 | -0.1235 [-0.1371, -0.1092] | +0.0518 [+0.0423, +0.0607] | yes |
| cascast_cascade_cfg1 − pysteps | max16 | 133 | 0.91 / 0.78 | -0.0197 [-0.0254, -0.0142] | +0.0157 [+0.0116, +0.0199] | yes |
| cascast_cascade_cfg1 − pysteps | max16 | 160 | 0.91 / 0.69 | -0.0213 [-0.0284, -0.0144] | +0.0164 [+0.0112, +0.0217] | yes |
| cascast_cascade_cfg1 − pysteps | max16 | 181 | 0.92 / 0.66 | -0.0046 [-0.0127, +0.0032] | +0.0276 [+0.0206, +0.0346] |  |
| cascast_cascade_cfg1 − pysteps | max16 | 219 | 0.91 / 0.55 | -0.0225 [-0.0342, -0.0117] | +0.0242 [+0.0144, +0.0336] | yes |
| cascast_det − earthformer | none | 133 | 0.91 / 0.81 | -0.0060 [-0.0087, -0.0034] | +0.0018 [-0.0005, +0.0040] |  |
| cascast_det − earthformer | none | 160 | 0.71 / 0.56 | -0.0170 [-0.0209, -0.0131] | +0.0016 [-0.0013, +0.0046] |  |
| cascast_det − earthformer | none | 181 | 0.62 / 0.47 | -0.0232 [-0.0283, -0.0182] | +0.0025 [-0.0015, +0.0065] |  |
| cascast_det − earthformer | none | 219 | 0.55 / 0.29 | -0.0356 [-0.0453, -0.0259] | +0.0092 [+0.0027, +0.0158] | yes |
| cascast_det − earthformer | avg4 | 133 | 0.97 / 0.87 | -0.0030 [-0.0059, -0.0002] | +0.0023 [-0.0002, +0.0048] |  |
| cascast_det − earthformer | avg4 | 160 | 0.79 / 0.62 | -0.0133 [-0.0175, -0.0090] | +0.0017 [-0.0015, +0.0051] |  |
| cascast_det − earthformer | avg4 | 181 | 0.71 / 0.53 | -0.0200 [-0.0260, -0.0141] | +0.0037 [-0.0009, +0.0083] |  |
| cascast_det − earthformer | avg4 | 219 | 0.67 / 0.35 | -0.0295 [-0.0416, -0.0171] | +0.0108 [+0.0031, +0.0184] | yes |
| cascast_det − earthformer | max4 | 74 | 0.87 / 0.84 | -0.0048 [-0.0063, -0.0033] | +0.0016 [+0.0002, +0.0030] | yes |
| cascast_det − earthformer | max4 | 133 | 0.69 / 0.62 | -0.0168 [-0.0198, -0.0137] | +0.0007 [-0.0012, +0.0027] |  |
| cascast_det − earthformer | max4 | 160 | 0.52 / 0.42 | -0.0280 [-0.0320, -0.0239] | +0.0014 [-0.0014, +0.0042] |  |
| cascast_det − earthformer | max4 | 181 | 0.45 / 0.36 | -0.0314 [-0.0361, -0.0270] | +0.0042 [+0.0009, +0.0076] | yes |
| cascast_det − earthformer | max4 | 219 | 0.38 / 0.21 | -0.0503 [-0.0582, -0.0419] | +0.0072 [+0.0020, +0.0124] | yes |
| cascast_det − earthformer | max16 | 16 | 0.91 / 0.87 | -0.0109 [-0.0125, -0.0094] | +0.0006 [-0.0003, +0.0014] |  |
| cascast_det − earthformer | max16 | 74 | 0.75 / 0.72 | -0.0150 [-0.0171, -0.0129] | +0.0005 [-0.0007, +0.0018] |  |
| cascast_det − earthformer | max16 | 133 | 0.53 / 0.49 | -0.0238 [-0.0272, -0.0202] | +0.0013 [-0.0004, +0.0031] |  |
| cascast_det − earthformer | max16 | 160 | 0.39 / 0.34 | -0.0315 [-0.0358, -0.0274] | +0.0014 [-0.0010, +0.0040] |  |
| cascast_det − earthformer | max16 | 181 | 0.34 / 0.28 | -0.0323 [-0.0371, -0.0280] | +0.0047 [+0.0013, +0.0079] | yes |
| cascast_det − earthformer | max16 | 219 | 0.28 / 0.17 | -0.0614 [-0.0695, -0.0536] | +0.0089 [+0.0044, +0.0136] | yes |
| cascast_cascade − earthformer (published) | max4 | 160 | 0.52 / 0.93 | +0.0124 [+0.0072, +0.0176] | -0.0199 [-0.0237, -0.0158] | yes |
| cascast_cascade − earthformer (published) | max4 | 181 | 0.45 / 0.95 | +0.0200 [+0.0131, +0.0267] | -0.0207 [-0.0258, -0.0156] | yes |
| cascast_cascade − earthformer (published) | max4 | 219 | 0.38 / 1.03 | +0.0104 [+0.0000, +0.0208] | -0.0348 [-0.0427, -0.0272] | yes |
| cascast_cascade − earthformer (published) | max16 | 74 | 0.75 / 0.97 | +0.0329 [+0.0286, +0.0368] | -0.0119 [-0.0144, -0.0095] | yes |
| cascast_cascade − earthformer (published) | max16 | 133 | 0.53 / 0.92 | +0.0942 [+0.0876, +0.1008] | -0.0080 [-0.0117, -0.0044] | yes |
| cascast_cascade − earthformer (published) | max16 | 219 | 0.28 / 1.05 | +0.1001 [+0.0863, +0.1129] | -0.0120 [-0.0197, -0.0037] | yes |
| cascast_cascade_cfg1 − earthformer | max16 | 74 | 0.75 / 0.91 | +0.0059 [+0.0021, +0.0092] | -0.0308 [-0.0334, -0.0282] | yes |
| cascast_cascade_cfg1 − earthformer | max16 | 133 | 0.53 / 0.78 | +0.0451 [+0.0398, +0.0504] | -0.0248 [-0.0280, -0.0218] | yes |
| cascast_cascade_cfg1 − earthformer | max16 | 160 | 0.39 / 0.69 | +0.0705 [+0.0645, +0.0764] | -0.0105 [-0.0149, -0.0062] | yes |
| cascast_cascade_cfg1 − earthformer | max16 | 219 | 0.28 / 0.55 | +0.0395 [+0.0289, +0.0502] | -0.0187 [-0.0256, -0.0110] | yes |
| cascast_cascade − cascast_det (published) | none | 219 | 0.29 / 1.09 | +0.0153 [+0.0049, +0.0254] | -0.0479 [-0.0545, -0.0409] | yes |
| cascast_cascade − cascast_det (published) | avg4 | 219 | 0.35 / 1.16 | +0.0061 [-0.0065, +0.0178] | -0.0442 [-0.0518, -0.0365] |  |
| cascast_cascade − cascast_det (published) | max4 | 133 | 0.62 / 0.96 | +0.0123 [+0.0080, +0.0166] | -0.0380 [-0.0406, -0.0354] | yes |
| cascast_cascade − cascast_det (published) | max4 | 160 | 0.42 / 0.93 | +0.0404 [+0.0346, +0.0460] | -0.0213 [-0.0245, -0.0179] | yes |
| cascast_cascade − cascast_det (published) | max4 | 181 | 0.36 / 0.95 | +0.0514 [+0.0445, +0.0577] | -0.0249 [-0.0291, -0.0204] | yes |
| cascast_cascade − cascast_det (published) | max4 | 219 | 0.21 / 1.03 | +0.0606 [+0.0500, +0.0707] | -0.0421 [-0.0482, -0.0360] | yes |
| cascast_cascade − cascast_det (published) | max16 | 74 | 0.72 / 0.97 | +0.0479 [+0.0431, +0.0522] | -0.0124 [-0.0151, -0.0097] | yes |
| cascast_cascade − cascast_det (published) | max16 | 133 | 0.49 / 0.92 | +0.1180 [+0.1107, +0.1251] | -0.0094 [-0.0127, -0.0062] | yes |
| cascast_cascade − cascast_det (published) | max16 | 219 | 0.17 / 1.05 | +0.1615 [+0.1480, +0.1739] | -0.0208 [-0.0282, -0.0131] | yes |
| cascast_cascade_cfg1 − cascast_det | max4 | 181 | 0.36 / 0.65 | +0.0112 [+0.0058, +0.0163] | -0.0447 [-0.0485, -0.0407] | yes |
| cascast_cascade_cfg1 − cascast_det | max4 | 219 | 0.21 / 0.51 | +0.0142 [+0.0047, +0.0239] | -0.0530 [-0.0585, -0.0471] | yes |
| cascast_cascade_cfg1 − cascast_det | max16 | 74 | 0.72 / 0.91 | +0.0209 [+0.0171, +0.0243] | -0.0313 [-0.0341, -0.0286] | yes |
| cascast_cascade_cfg1 − cascast_det | max16 | 133 | 0.49 / 0.78 | +0.0688 [+0.0632, +0.0745] | -0.0262 [-0.0291, -0.0233] | yes |
| cascast_cascade_cfg1 − cascast_det | max16 | 160 | 0.34 / 0.69 | +0.1021 [+0.0962, +0.1077] | -0.0119 [-0.0161, -0.0078] | yes |
| cascast_cascade_cfg1 − cascast_det | max16 | 181 | 0.28 / 0.66 | +0.1088 [+0.1024, +0.1153] | -0.0046 [-0.0093, +0.0002] |  |
| cascast_cascade_cfg1 − cascast_det | max16 | 219 | 0.17 / 0.55 | +0.1009 [+0.0909, +0.1111] | -0.0276 [-0.0334, -0.0213] | yes |

## Published contrasts

| b − a | pooling | τ | raw [95% CI] | calibrated [95% CI] |
|---|---|---|---|---|
| earthformer − persistence | none | 16 | +0.1552 [+0.1490, +0.1615] | +0.1589 [+0.1519, +0.1660] |
| earthformer − persistence | none | 74 | +0.2192 [+0.2117, +0.2273] | +0.2180 [+0.2105, +0.2260] |
| earthformer − persistence | none | 133 | +0.2034 [+0.1957, +0.2111] | +0.2090 [+0.2014, +0.2169] |
| earthformer − persistence | none | 160 | +0.1891 [+0.1768, +0.2010] | +0.1922 [+0.1809, +0.2032] |
| earthformer − persistence | none | 181 | +0.1819 [+0.1658, +0.1987] | +0.1945 [+0.1804, +0.2095] |
| earthformer − persistence | none | 219 | +0.1262 [+0.1060, +0.1466] | +0.1411 [+0.1224, +0.1595] |
| earthformer − persistence | avg4 | 16 | +0.1586 [+0.1522, +0.1652] | +0.1693 [+0.1628, +0.1761] |
| earthformer − persistence | avg4 | 74 | +0.2238 [+0.2157, +0.2324] | +0.2239 [+0.2158, +0.2325] |
| earthformer − persistence | avg4 | 133 | +0.2138 [+0.2055, +0.2222] | +0.2149 [+0.2067, +0.2234] |
| earthformer − persistence | avg4 | 160 | +0.2091 [+0.1964, +0.2215] | +0.2095 [+0.1973, +0.2214] |
| earthformer − persistence | avg4 | 181 | +0.2044 [+0.1875, +0.2223] | +0.2106 [+0.1950, +0.2273] |
| earthformer − persistence | avg4 | 219 | +0.1437 [+0.1214, +0.1649] | +0.1508 [+0.1295, +0.1714] |
| earthformer − persistence | max4 | 16 | +0.1525 [+0.1469, +0.1584] | +0.1509 [+0.1451, +0.1569] |
| earthformer − persistence | max4 | 74 | +0.1931 [+0.1861, +0.1996] | +0.2025 [+0.1958, +0.2094] |
| earthformer − persistence | max4 | 133 | +0.1684 [+0.1608, +0.1764] | +0.2039 [+0.1964, +0.2118] |
| earthformer − persistence | max4 | 160 | +0.1442 [+0.1328, +0.1556] | +0.1788 [+0.1690, +0.1887] |
| earthformer − persistence | max4 | 181 | +0.1395 [+0.1249, +0.1545] | +0.1813 [+0.1689, +0.1937] |
| earthformer − persistence | max4 | 219 | +0.1058 [+0.0867, +0.1262] | +0.1484 [+0.1321, +0.1649] |
| earthformer − persistence | avg16 | 16 | +0.1575 [+0.1508, +0.1642] | +0.1655 [+0.1590, +0.1723] |
| earthformer − persistence | avg16 | 74 | +0.2314 [+0.2215, +0.2417] | +0.2335 [+0.2235, +0.2439] |
| earthformer − persistence | avg16 | 133 | +0.2407 [+0.2274, +0.2536] | +0.2463 [+0.2334, +0.2592] |
| earthformer − persistence | avg16 | 160 | +0.2537 [+0.2337, +0.2738] | +0.2644 [+0.2440, +0.2844] |
| earthformer − persistence | avg16 | 181 | +0.2580 [+0.2278, +0.2892] | +0.2627 [+0.2315, +0.2945] |
| earthformer − persistence | avg16 | 219 | +0.1545 [+0.1148, +0.1891] | +0.1673 [+0.1241, +0.2042] |
| earthformer − persistence | max16 | 16 | +0.0873 [+0.0824, +0.0926] | +0.0964 [+0.0920, +0.1010] |
| earthformer − persistence | max16 | 74 | +0.0801 [+0.0737, +0.0867] | +0.1275 [+0.1220, +0.1332] |
| earthformer − persistence | max16 | 133 | +0.0320 [+0.0228, +0.0415] | +0.1428 [+0.1352, +0.1502] |
| earthformer − persistence | max16 | 160 | -0.0014 [-0.0135, +0.0112] | +0.1215 [+0.1120, +0.1310] |
| earthformer − persistence | max16 | 181 | +0.0065 [-0.0075, +0.0209] | +0.1176 [+0.1057, +0.1292] |
| earthformer − persistence | max16 | 219 | +0.0156 [-0.0036, +0.0359] | +0.1230 [+0.1077, +0.1373] |
| cascast_cascade − earthformer | none | 16 | -0.0151 [-0.0184, -0.0122] | -0.0204 [-0.0242, -0.0171] |
| cascast_cascade − earthformer | none | 74 | -0.0363 [-0.0381, -0.0344] | -0.0361 [-0.0378, -0.0342] |
| cascast_cascade − earthformer | none | 133 | -0.0386 [-0.0418, -0.0355] | -0.0461 [-0.0492, -0.0430] |
| cascast_cascade − earthformer | none | 160 | -0.0313 [-0.0356, -0.0269] | -0.0354 [-0.0394, -0.0311] |
| cascast_cascade − earthformer | none | 181 | -0.0264 [-0.0325, -0.0202] | -0.0408 [-0.0456, -0.0358] |
| cascast_cascade − earthformer | none | 219 | -0.0203 [-0.0298, -0.0104] | -0.0387 [-0.0484, -0.0292] |
| cascast_cascade − earthformer | avg4 | 16 | -0.0162 [-0.0195, -0.0134] | -0.0269 [-0.0303, -0.0242] |
| cascast_cascade − earthformer | avg4 | 74 | -0.0318 [-0.0336, -0.0299] | -0.0325 [-0.0343, -0.0306] |
| cascast_cascade − earthformer | avg4 | 133 | -0.0362 [-0.0396, -0.0329] | -0.0409 [-0.0444, -0.0376] |
| cascast_cascade − earthformer | avg4 | 160 | -0.0343 [-0.0388, -0.0298] | -0.0348 [-0.0391, -0.0304] |
| cascast_cascade − earthformer | avg4 | 181 | -0.0315 [-0.0377, -0.0251] | -0.0386 [-0.0438, -0.0332] |
| cascast_cascade − earthformer | avg4 | 219 | -0.0234 [-0.0342, -0.0127] | -0.0334 [-0.0450, -0.0224] |
| cascast_cascade − earthformer | max4 | 16 | -0.0299 [-0.0329, -0.0273] | -0.0296 [-0.0328, -0.0270] |
| cascast_cascade − earthformer | max4 | 74 | -0.0246 [-0.0267, -0.0225] | -0.0325 [-0.0343, -0.0307] |
| cascast_cascade − earthformer | max4 | 133 | -0.0045 [-0.0088, -0.0004] | -0.0373 [-0.0404, -0.0344] |
| cascast_cascade − earthformer | max4 | 160 | +0.0124 [+0.0072, +0.0176] | -0.0199 [-0.0237, -0.0158] |
| cascast_cascade − earthformer | max4 | 181 | +0.0200 [+0.0131, +0.0267] | -0.0207 [-0.0258, -0.0156] |
| cascast_cascade − earthformer | max4 | 219 | +0.0104 [+0.0000, +0.0208] | -0.0348 [-0.0427, -0.0272] |
| cascast_cascade − earthformer | avg16 | 16 | -0.0172 [-0.0207, -0.0144] | -0.0249 [-0.0282, -0.0222] |
| cascast_cascade − earthformer | avg16 | 74 | -0.0150 [-0.0171, -0.0127] | -0.0176 [-0.0198, -0.0152] |
| cascast_cascade − earthformer | avg16 | 133 | -0.0225 [-0.0273, -0.0173] | -0.0308 [-0.0357, -0.0258] |
| cascast_cascade − earthformer | avg16 | 160 | -0.0163 [-0.0237, -0.0075] | -0.0256 [-0.0333, -0.0171] |
| cascast_cascade − earthformer | avg16 | 181 | -0.0150 [-0.0271, -0.0025] | -0.0216 [-0.0356, -0.0074] |
| cascast_cascade − earthformer | avg16 | 219 | -0.0093 [-0.0383, +0.0185] | -0.0296 [-0.0650, +0.0029] |
| cascast_cascade − earthformer | max16 | 16 | -0.0129 [-0.0160, -0.0103] | -0.0219 [-0.0245, -0.0197] |
| cascast_cascade − earthformer | max16 | 74 | +0.0329 [+0.0286, +0.0368] | -0.0119 [-0.0144, -0.0095] |
| cascast_cascade − earthformer | max16 | 133 | +0.0942 [+0.0876, +0.1008] | -0.0080 [-0.0117, -0.0044] |
| cascast_cascade − earthformer | max16 | 160 | +0.1268 [+0.1193, +0.1343] | +0.0122 [+0.0073, +0.0170] |
| cascast_cascade − earthformer | max16 | 181 | +0.1317 [+0.1228, +0.1399] | +0.0240 [+0.0178, +0.0302] |
| cascast_cascade − earthformer | max16 | 219 | +0.1001 [+0.0863, +0.1129] | -0.0120 [-0.0197, -0.0037] |
| cascast_cascade − cascast_det | none | 16 | -0.0275 [-0.0307, -0.0249] | -0.0249 [-0.0285, -0.0218] |
| cascast_cascade − cascast_det | none | 74 | -0.0383 [-0.0396, -0.0370] | -0.0393 [-0.0407, -0.0378] |
| cascast_cascade − cascast_det | none | 133 | -0.0326 [-0.0354, -0.0297] | -0.0478 [-0.0502, -0.0454] |
| cascast_cascade − cascast_det | none | 160 | -0.0143 [-0.0191, -0.0097] | -0.0370 [-0.0400, -0.0339] |
| cascast_cascade − cascast_det | none | 181 | -0.0032 [-0.0097, +0.0034] | -0.0433 [-0.0474, -0.0391] |
| cascast_cascade − cascast_det | none | 219 | +0.0153 [+0.0049, +0.0254] | -0.0479 [-0.0545, -0.0409] |
| cascast_cascade − cascast_det | avg4 | 16 | -0.0288 [-0.0321, -0.0262] | -0.0299 [-0.0333, -0.0273] |
| cascast_cascade − cascast_det | avg4 | 74 | -0.0351 [-0.0364, -0.0337] | -0.0362 [-0.0376, -0.0347] |
| cascast_cascade − cascast_det | avg4 | 133 | -0.0332 [-0.0360, -0.0303] | -0.0432 [-0.0456, -0.0408] |
| cascast_cascade − cascast_det | avg4 | 160 | -0.0210 [-0.0257, -0.0163] | -0.0366 [-0.0397, -0.0333] |
| cascast_cascade − cascast_det | avg4 | 181 | -0.0115 [-0.0185, -0.0045] | -0.0423 [-0.0465, -0.0378] |
| cascast_cascade − cascast_det | avg4 | 219 | +0.0061 [-0.0065, +0.0178] | -0.0442 [-0.0518, -0.0365] |
| cascast_cascade − cascast_det | max4 | 16 | -0.0324 [-0.0354, -0.0299] | -0.0329 [-0.0359, -0.0305] |
| cascast_cascade − cascast_det | max4 | 74 | -0.0197 [-0.0215, -0.0180] | -0.0342 [-0.0359, -0.0325] |
| cascast_cascade − cascast_det | max4 | 133 | +0.0123 [+0.0080, +0.0166] | -0.0380 [-0.0406, -0.0354] |
| cascast_cascade − cascast_det | max4 | 160 | +0.0404 [+0.0346, +0.0460] | -0.0213 [-0.0245, -0.0179] |
| cascast_cascade − cascast_det | max4 | 181 | +0.0514 [+0.0445, +0.0577] | -0.0249 [-0.0291, -0.0204] |
| cascast_cascade − cascast_det | max4 | 219 | +0.0606 [+0.0500, +0.0707] | -0.0421 [-0.0482, -0.0360] |
| cascast_cascade − cascast_det | avg16 | 16 | -0.0262 [-0.0296, -0.0237] | -0.0259 [-0.0292, -0.0235] |
| cascast_cascade − cascast_det | avg16 | 74 | -0.0221 [-0.0235, -0.0205] | -0.0225 [-0.0239, -0.0209] |
| cascast_cascade − cascast_det | avg16 | 133 | -0.0295 [-0.0329, -0.0260] | -0.0322 [-0.0355, -0.0290] |
| cascast_cascade − cascast_det | avg16 | 160 | -0.0281 [-0.0347, -0.0210] | -0.0304 [-0.0351, -0.0252] |
| cascast_cascade − cascast_det | avg16 | 181 | -0.0272 [-0.0397, -0.0151] | -0.0400 [-0.0493, -0.0300] |
| cascast_cascade − cascast_det | avg16 | 219 | -0.0233 [-0.0485, -0.0019] | -0.0415 [-0.0627, -0.0196] |
| cascast_cascade − cascast_det | max16 | 16 | -0.0020 [-0.0054, +0.0010] | -0.0225 [-0.0249, -0.0204] |
| cascast_cascade − cascast_det | max16 | 74 | +0.0479 [+0.0431, +0.0522] | -0.0124 [-0.0151, -0.0097] |
| cascast_cascade − cascast_det | max16 | 133 | +0.1180 [+0.1107, +0.1251] | -0.0094 [-0.0127, -0.0062] |
| cascast_cascade − cascast_det | max16 | 160 | +0.1583 [+0.1502, +0.1661] | +0.0108 [+0.0064, +0.0154] |
| cascast_cascade − cascast_det | max16 | 181 | +0.1640 [+0.1557, +0.1719] | +0.0193 [+0.0140, +0.0246] |
| cascast_cascade − cascast_det | max16 | 219 | +0.1615 [+0.1480, +0.1739] | -0.0208 [-0.0282, -0.0131] |
