# Feature ablation: pooled OOF vs LOCO (dev, frozen blocker, tau 0.75)

France share of test S1 f = 0.1498. expected = (1-f)*pooled + f*mean(LOCO US, LOCO India-Latin).
Paired bootstrap over S1 (2000 replicates) of expected(variant) - expected(current).

| variant | pooled | pooled US | pooled IN | LOCO US | LOCO IN-Latin | LOCO IN-all | expected | d vs current [95% CI] | P(better) |
|---|---|---|---|---|---|---|---|---|---|
| current (/lnN) | 0.9605 | 0.9716 | 0.9436 | 0.9527 | 0.9098 | 0.8436 | 0.9562 | +0.0000 [+0.0000, +0.0000] | 0.00 |
| raw idf sums | 0.9598 | 0.9712 | 0.9423 | 0.9476 | 0.9076 | 0.8431 | 0.9550 | -0.0012 [-0.0017, -0.0007] | 0.00 |
| within-country pct | 0.9600 | 0.9716 | 0.9423 | 0.9483 | 0.9044 | 0.8391 | 0.9550 | -0.0012 [-0.0018, -0.0006] | 0.00 |
| -idf feats | 0.9538 | 0.9675 | 0.9328 | 0.9519 | 0.9009 | 0.8472 | 0.9497 | -0.0065 [-0.0073, -0.0056] | 0.00 |
| -addr/len struct | 0.9579 | 0.9692 | 0.9405 | 0.9491 | 0.9016 | 0.8392 | 0.9530 | -0.0032 [-0.0039, -0.0025] | 0.00 |
| -postal | 0.9600 | 0.9712 | 0.9429 | 0.9529 | 0.9094 | 0.8424 | 0.9557 | -0.0005 [-0.0009, -0.0000] | 0.02 |
| -script | 0.9597 | 0.9714 | 0.9417 | 0.9516 | 0.9121 | 0.8457 | 0.9555 | -0.0007 [-0.0012, -0.0002] | 0.00 |
| -idf -struct | 0.9471 | 0.9623 | 0.9239 | 0.9393 | 0.8837 | 0.8603 | 0.9418 | -0.0144 [-0.0155, -0.0133] | 0.00 |
| pct -struct | 0.9578 | 0.9696 | 0.9396 | 0.9414 | 0.9014 | 0.8327 | 0.9523 | -0.0039 [-0.0046, -0.0032] | 0.00 |

Best by expected: **current (/lnN)** (d +0.0000, 95% CI [+0.0000, +0.0000]). Switch only if the CI excludes 0.
