# shallow-resnet

> Does a skip connection help in shallow ResNets? A parameter-matched, multi-seed ablation at 8 and 20 layers on CIFAR-10 (PyTorch).

Does the value of a residual (skip) connection depend on network depth? This repository contains a
parameter-matched, multi-seed ablation of skip connections at 8 and 20 layers on CIFAR-10, with batch
normalization on throughout, together with the report, the code, the raw per-run results, and the scripts that
turn those results into every number and figure in the report.

**Report:** [`report/report_en.md`](report/report_en.md)

## Main result

Cosine learning-rate schedule (primary protocol), final-epoch test accuracy, 3 seeds per cell, 10 epochs:

| Depth | Residual ON | Residual OFF | ON − OFF (95% CI) |
|---|---|---|---|
| 8  | 74.38 ± 1.20 | 74.80 ± 0.28 | −0.42 [−3.22, +2.38] |
| 20 | 82.63 ± 0.24 | 80.15 ± 0.37 | **+2.48 [+1.73, +3.24]** |

Mean ± sample SD in percent. The depth-by-residual interaction is significant (p = 0.005 with pooled variance,
p = 0.036 without). The residual-on and residual-off networks have exactly the same parameter count (75,290 at
depth 8; 269,722 at depth 20) because shortcuts are parameter-free.

Read the report's Limitations before quoting these numbers: only two depths, BN is never switched off, three seeds
give limited power, and ten epochs is far from a converged schedule.

## Repository layout

```
report/                    report_en.md and its figures
notebooks/                 BN_Residual_Depth_Ablation.ipynb  (the experiment; needs a GPU)
results/                   raw per-run results (JSON) and the figures the notebook produced
  s8_residual_depth_results.json          constant learning rate (first stage)
  s8_residual_depth_results_cosine.json   cosine learning rate (primary protocol)
  notebook_figures/                       figures written by the notebook
  pilot_superseded/                       an abandoned early pilot; not used in any analysis
analysis/                  stats.py and make_figures.py: recompute every statistic and figure from results/
related_work/              literature notes and the list of papers used
docs/                      a personal narrative of how the project developed (Chinese)
archive/                   an earlier table-generation script; superseded, kept for the record
```

See [`results/README.md`](results/README.md) for the JSON schema.

## Reproducing

**Statistics and figures from the stored results** (no GPU needed):

```bash
pip install -r requirements.txt      # numpy and matplotlib are enough for this step
python analysis/stats.py             # prints every statistic quoted in the report
python analysis/make_figures.py      # rewrites report/figures/fig2-fig5
```

**Re-running the experiments** (GPU):

1. Open `notebooks/BN_Residual_Depth_Ablation.ipynb` in Google Colab with a GPU runtime (or locally with PyTorch and CUDA).
2. Run the cells in order: Setup, architecture, `run_single`, then the experiment cell. Set `SCHEDULE = 'constant'`
   (first stage) or `'cosine'` (primary protocol) in the experiment cell.
3. On Colab the notebook mounts Google Drive and writes results, checkpoints and figures there
   (`RESULTS_DIR`). Every finished run is appended to a JSON file and runs already present are skipped, so an
   interrupted session resumes. To reuse the stored results, copy the two JSON files from `results/` into that folder.
4. CIFAR-10 is downloaded automatically by torchvision and is not included in this repository.

Seeds are fixed and cuDNN is set to deterministic mode; epoch 1 was bit-identical between the two executions of the
experiment. Exact reproduction on a different GPU or library version is not guaranteed.

## Design in brief

- **Networks:** the CIFAR-10 ResNet of He et al. (2016), depth = 6n + 2, three stages of widths 16/32/64, global
  average pooling. The residual-off network is the same network with the shortcut removed. Width/stride changes
  use option-A shortcuts (subsample + zero-pad), which add no parameters.
- **Recipes:** Adam, initial learning rate 1e-3, weight decay 1e-4, batch size 64, 10 epochs, random-crop and
  horizontal-flip augmentation. Constant learning rate (first stage; reports the epoch with the lowest test loss) or
  cosine annealing (primary protocol; reports the final epoch, so no epoch is selected).
- **Runs:** depth ∈ {8, 20} × residual ∈ {on, off} × seeds {2026, 2027, 2028}, under each recipe (24 runs).

## Provenance and attribution

- The helper functions in the notebook (`train`, `evaluate`, `train_model`, accuracy) are adapted from the starter
  code of NCCU Deep Learning Programming Assignment #2, of which this project is an independent extension. The
  coursework itself is not included.
- The papers listed in [`related_work/papers.md`](related_work/papers.md) are not redistributed; fetch them from arXiv.
- No license has been chosen for this repository yet.
