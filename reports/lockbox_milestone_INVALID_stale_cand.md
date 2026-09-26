# Lockbox milestone: baseline (stage 1, tau 0.75)

tau 0.75; models frozen on dev folds; lockbox 45,000 S1. Bootstrap 95% CI over S1 (2000).
test-mix = US/India weighted by test_source1 counts {'India': 809986, 'US': 663106}.
France estimate (LOCO Latin view, mean of US-LOCO 0.9527 and India-Latin-LOCO 0.9098): 0.9313

| set | country | n | macro_f05 | ci_lo | ci_hi | singleton_f05 | multi_f05 | micro_p | micro_r | oracle | empty_share_pred | singleton_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lockbox | ALL | 45000 | 0.0747 | 0.0722 | 0.0771 | 0.9976 | 0.0201 | 0.9861 | 0.0196 | 0.0753 | 0.9803 | 0.0558 |
| lockbox | India | 18012 | 0.0744 | 0.0708 | 0.0783 | 0.9980 | 0.0197 | 0.9827 | 0.0191 | 0.0751 | 0.9805 | 0.0559 |
| lockbox | US | 26988 | 0.0748 | 0.0718 | 0.0780 | 0.9973 | 0.0203 | 0.9883 | 0.0199 | 0.0755 | 0.9803 | 0.0558 |
| lockbox | test-mix | 45000 | 0.0746 | 0.0721 | 0.0771 |  |  |  |  |  |  |  |
| dev OOF | ALL | 22133 | 0.9605 | 0.9589 | 0.9621 | 0.9549 | 0.9609 | 0.9892 | 0.9159 | 0.9846 | 0.0622 | 0.0561 |
| dev OOF | India | 8751 | 0.9436 | 0.9405 | 0.9466 | 0.9420 | 0.9437 | 0.9868 | 0.8824 | 0.9707 | 0.0667 | 0.0552 |
| dev OOF | US | 13382 | 0.9716 | 0.9700 | 0.9733 | 0.9631 | 0.9722 | 0.9907 | 0.9379 | 0.9937 | 0.0593 | 0.0566 |
| dev OOF | test-mix | 22133 | 0.9562 | 0.9544 | 0.9582 |  |  |  |  |  |  |  |

