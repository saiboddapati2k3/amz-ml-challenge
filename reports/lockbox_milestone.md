# Lockbox milestone: baseline (stage 1, tau 0.75)

tau 0.75; models frozen on dev folds; lockbox 45,000 S1. Bootstrap 95% CI over S1 (2000).
test-mix = US/India weighted by test_source1 counts {'US': 663106, 'India': 809986}.
France estimate (LOCO Latin view, mean of US-LOCO 0.9527 and India-Latin-LOCO 0.9098): 0.9313

| set | country | n | macro_f05 | ci_lo | ci_hi | singleton_f05 | multi_f05 | micro_p | micro_r | oracle | empty_share_pred | singleton_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lockbox | ALL | 45000 | 0.9603 | 0.9591 | 0.9614 | 0.9499 | 0.9609 | 0.9896 | 0.9158 | 0.9845 | 0.0621 | 0.0558 |
| lockbox | India | 18012 | 0.9434 | 0.9410 | 0.9455 | 0.9464 | 0.9432 | 0.9870 | 0.8824 | 0.9707 | 0.0681 | 0.0559 |
| lockbox | US | 26988 | 0.9716 | 0.9703 | 0.9728 | 0.9522 | 0.9727 | 0.9912 | 0.9382 | 0.9938 | 0.0580 | 0.0558 |
| lockbox | test-mix | 45000 | 0.9561 | 0.9547 | 0.9574 |  |  |  |  |  |  |  |
| dev OOF | ALL | 22133 | 0.9605 | 0.9589 | 0.9621 | 0.9549 | 0.9609 | 0.9892 | 0.9159 | 0.9846 | 0.0622 | 0.0561 |
| dev OOF | India | 8751 | 0.9436 | 0.9405 | 0.9466 | 0.9420 | 0.9437 | 0.9868 | 0.8824 | 0.9707 | 0.0667 | 0.0552 |
| dev OOF | US | 13382 | 0.9716 | 0.9700 | 0.9733 | 0.9631 | 0.9722 | 0.9907 | 0.9379 | 0.9937 | 0.0593 | 0.0566 |
| dev OOF | test-mix | 22133 | 0.9562 | 0.9543 | 0.9582 |  |  |  |  |  |  |  |

# Lockbox milestone: baseline (stage 1, tau 0.75)

tau 0.75; models frozen on dev folds; lockbox 45,000 S1. Bootstrap 95% CI over S1 (2000).
test-mix = US/India weighted by test_source1 counts {'US': 663106, 'India': 809986}.
France estimate (LOCO Latin view, mean of US-LOCO 0.9527 and India-Latin-LOCO 0.9098): 0.9313

| set | country | n | macro_f05 | ci_lo | ci_hi | singleton_f05 | multi_f05 | micro_p | micro_r | oracle | empty_share_pred | singleton_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lockbox | ALL | 45000 | 0.9603 | 0.9591 | 0.9614 | 0.9499 | 0.9609 | 0.9896 | 0.9158 | 0.9845 | 0.0621 | 0.0558 |
| lockbox | India | 18012 | 0.9434 | 0.9410 | 0.9455 | 0.9464 | 0.9432 | 0.9870 | 0.8824 | 0.9707 | 0.0681 | 0.0559 |
| lockbox | US | 26988 | 0.9716 | 0.9703 | 0.9728 | 0.9522 | 0.9727 | 0.9912 | 0.9382 | 0.9938 | 0.0580 | 0.0558 |
| lockbox | test-mix | 45000 | 0.9561 | 0.9547 | 0.9574 |  |  |  |  |  |  |  |
| dev OOF | ALL | 22133 | 0.9605 | 0.9589 | 0.9621 | 0.9549 | 0.9609 | 0.9892 | 0.9159 | 0.9846 | 0.0622 | 0.0561 |
| dev OOF | India | 8751 | 0.9436 | 0.9405 | 0.9466 | 0.9420 | 0.9437 | 0.9868 | 0.8824 | 0.9707 | 0.0667 | 0.0552 |
| dev OOF | US | 13382 | 0.9716 | 0.9700 | 0.9733 | 0.9631 | 0.9722 | 0.9907 | 0.9379 | 0.9937 | 0.0593 | 0.0566 |
| dev OOF | test-mix | 22133 | 0.9562 | 0.9543 | 0.9582 |  |  |  |  |  |  |  |

