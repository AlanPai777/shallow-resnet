# Results

## Files

| File | What it is |
|---|---|
| `s8_residual_depth_results.json` | 12 runs, **constant** learning rate (first stage). |
| `s8_residual_depth_results_cosine.json` | 12 runs, **cosine** learning rate (primary protocol). |
| `notebook_figures/` | Figures the notebook wrote. `interaction_cosine.png`, `learning_curves_cosine.png` and `delta_by_schedule.png` come from the cosine run; `gradient_norms_depth8.png` / `gradient_norms_depth20.png` are schedule-independent; `residual_depth_delta.png` is from an earlier notebook version and shows the legacy metric of the constant-LR stage. |
| `pilot_superseded/s8_experimentA_results.json` | 5 runs (depth 8, residual on, seeds 2026–2030, 8 epochs) from an abandoned early design. Not used in any analysis. |

The per-layer gradient values behind the gradient figures were not saved, only the figures.

## JSON schema (one object per run)

| Field | Meaning |
|---|---|
| `n`, `depth` | blocks per stage and depth = 6n + 2 (n = 1, 3 → depth 8, 20) |
| `use_bn`, `use_residual` | switches; `use_bn` is always `true` in these results |
| `seed` | controls weight initialization and training-data order |
| `epochs`, `num_params` | 10; 75,290 (depth 8) or 269,722 (depth 20), identical for residual on/off |
| `train_losses`, `train_accuracies` | per-epoch training metrics (running average over the epoch, augmentation on, BN in training mode) |
| `valid_losses`, `valid_accuracies` | per-epoch metrics on the **CIFAR-10 test set** (there is no separate validation split) |
| `best_epoch`, `test_acc`, `test_loss` | the **legacy** metric: the epoch with the lowest test loss and its accuracy/loss. This selects on the test set; do not quote it in isolation |
| `final_test_acc`, `final_test_loss`, `lrs`, `schedule` | cosine file only: metrics after the last epoch (no selection), the learning rate used in each epoch, and `"cosine"` |

For the constant-LR file the final-epoch accuracy is `valid_accuracies[-1]`; the report's primary metric for both
files is the final-epoch accuracy. Accuracies are fractions in [0, 1].

`analysis/stats.py` reads both files and reproduces every number in the report.
