# Cell Type Annotation Benchmark Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** turn the current scGPT release into a cell type annotation benchmark that can compete on a real label-transfer task rather than only on small in-domain splits.

**Architecture:** keep the current repository layout, but add a new cross-dataset annotation track with a paired reference/query dataset, harmonized labels, stronger annotation baselines, and per-run analysis artifacts. The first target should be a real immune-cell label-transfer benchmark; the current best candidate is the CellTypist gut-atlas pair exposed in the official tutorial (`Elmentaite` reference → `James` query), with a fallback PBMC annotation track for smoke tests and reproducibility. The benchmark runner will orchestrate dataset loading, seed control, baseline execution, scGPT fine-tuning, and per-run analysis summaries.

**Tech Stack:** Python, scanpy, anndata, scikit-learn, torch, scGPT, matplotlib/seaborn, optional CellTypist for a strong annotation baseline.

---

## Design decision

### Primary benchmark target
Use **cross-dataset cell type annotation / label transfer** as the main leaderboard-style task.

Recommended first target:
- reference: CellTypist gut atlas `Elmentaite`
- query: CellTypist gut atlas `James`
- reference labels: `Integrated_05`
- query labels: `cell_type`

Why this target:
- it is a real label-transfer task, not a small in-domain split
- the tutorial already exposes the dataset URLs and label columns
- it is more likely to separate representation learning methods from simple PCA baselines
- it is much closer to a publishable annotation benchmark than the current PBMC toy split

### Fallback / smoke target
Keep a lightweight PBMC annotation smoke test in the repo so the pipeline still has a quick sanity check:
- `pbmc68k_reduced` as a fast in-domain annotation test
- only used for smoke validation, not for the main ranking

### Ranking metrics
For the main annotation task:
- primary: `macro_f1`
- secondary: `accuracy`
- supporting: `balanced_accuracy`
- auxiliary for clustering-style methods: `ARI`, `NMI`

---

## Task 1: Add a paired dataset registry for annotation tasks

**Objective:** introduce a dataset abstraction that can represent a reference/query pair, label columns, and gene-harmonization rules.

**Files:**
- Modify: `experiments/datasets.py`
- Create: `experiments/annotation_bench.py`
- Modify: `real_pbmc68k_finetune.py`
- Modify: `benchmark_baselines.py`
- Modify: `benchmark_runner.py`
- Modify: `configs/benchmark.yaml`

**Step 1: Write the failing test / validation probe**

Add a tiny loader probe that ensures the registry can describe:
- reference dataset name
- query dataset name
- reference label key
- query label key
- a shared gene intersection strategy

Validation command:
```bash
python -m py_compile experiments/datasets.py real_pbmc68k_finetune.py benchmark_baselines.py benchmark_runner.py
```
Expected: pass after implementation.

**Step 2: Implement the registry**

Add a `PairedDatasetSpec` dataclass that includes:
- `reference_name`
- `query_name`
- `reference_loader`
- `query_loader`
- `reference_label_key`
- `query_label_key`
- `description`
- optional `shared_gene_policy`

Add a new entry for the CellTypist gut atlas pair.

**Step 3: Verify the loaded metadata**

Run a small script that prints:
- number of cells in each side
- number of shared genes
- unique label count in reference/query

Expected: non-empty intersection and sane label cardinalities.

**Step 4: Commit**

```bash
git add experiments/datasets.py experiments/annotation_bench.py real_pbmc68k_finetune.py benchmark_baselines.py benchmark_runner.py configs/benchmark.yaml
git commit -m "feat: add paired cell annotation dataset registry"
```

---

## Task 2: Refactor scGPT fine-tuning into a real reference→query annotation flow

**Objective:** train on a reference dataset and evaluate on a held-out query dataset with a shared vocabulary.

**Files:**
- Modify: `real_pbmc68k_finetune.py`
- Modify: `experiments/datasets.py`
- Modify: `experiments/metrics.py`
- Modify: `experiments/analysis.py`
- Create: `experiments/label_transfer.py`

**Step 1: Write failing checks**

Add a minimal probe for the new CLI:
```bash
python real_pbmc68k_finetune.py --help
python real_pbmc68k_finetune.py --task label_transfer --dataset gutatlas_transfer --seed 0 --dry-run
```
Expected: new CLI options are exposed; dry-run exits after dataset/load validation.

**Step 2: Implement the refactor**

Move from the current single-dataset train/test split to:
- reference train/val split inside the reference dataset
- query set used only for final evaluation
- gene intersection across reference and query before tokenization
- single shared vocabulary built from intersected genes

Add outputs to the run directory:
- `reference_preprocessed.h5ad`
- `query_preprocessed.h5ad`
- `analysis_input.h5ad`
- `label_map.json`
- `summary.json`

**Step 3: Verify the data flow**

Run a smoke invocation and check the log prints:
- reference cells / query cells
- shared gene count
- number of classes after harmonization
- best checkpoint path

Expected: no shape mismatch in tokenization or classifier head.

**Step 4: Commit**

```bash
git add real_pbmc68k_finetune.py experiments/datasets.py experiments/metrics.py experiments/analysis.py experiments/label_transfer.py
git commit -m "feat: refactor scGPT for cross-dataset annotation"
```

---

## Task 3: Add stronger annotation baselines

**Objective:** compare scGPT against baselines that are actually relevant for cell annotation, not only generic PCA clustering.

**Files:**
- Modify: `benchmark_baselines.py`
- Modify: `benchmark_runner.py`
- Modify: `experiments/metrics.py`
- Modify: `README.md`

**Baselines to add first:**
- `celltypist_builtin` — pretrained CellTypist immune model
- `pca_logreg` — supervised baseline
- `pca_svm` — supervised baseline
- `pca_knn` — nearest-neighbor baseline
- `majority_class` — sanity floor
- keep `leiden` only as an auxiliary clustering baseline

**Step 1: Write the failing baseline summary test**

Validation command:
```bash
python benchmark_baselines.py --dataset gutatlas_transfer --methods celltypist_builtin pca_logreg pca_svm pca_knn majority_class --seed 0
```
Expected: currently fails until the new loader and baselines are wired up.

**Step 2: Implement baseline plumbing**

Each baseline should output:
- `metrics.json`
- `analysis_input.h5ad`
- `pred_label`
- `true_label`
- optional confidence scores

**Step 3: Verify baseline ordering**

Make sure the new baseline table reports:
- `macro_f1`
- `accuracy`
- `balanced_accuracy`
- `ARI`
- `NMI`

**Step 4: Commit**

```bash
git add benchmark_baselines.py benchmark_runner.py experiments/metrics.py README.md
git commit -m "feat: add annotation baselines"
```

---

## Task 4: Expand per-run analysis for leaderboard-style reporting

**Objective:** generate the artifacts needed to judge annotation quality, not just classification accuracy.

**Files:**
- Modify: `experiments/analysis.py`
- Modify: `benchmark_analysis.py`
- Modify: `benchmark_runner.py`
- Modify: `README.md`

**Artifacts to generate:**
- UMAP colored by true label and predicted label
- confusion matrix
- per-label precision/recall/F1 table
- marker genes for true and predicted groups
- summary JSON with primary metrics
- aggregated CSV over seeds

**Step 1: Write validation probes**

Run:
```bash
python benchmark_analysis.py --help
python benchmark_runner.py --help
```
Expected: both CLIs show the new annotation-related options.

**Step 2: Implement the analysis outputs**

For each run, save:
- `analysis/umap_true_label.png`
- `analysis/umap_pred_label.png`
- `analysis/confusion_matrix.csv`
- `analysis/classification_report.csv`
- `analysis/marker_true.csv`
- `analysis/marker_pred.csv`

**Step 3: Verify singleton-label safety**

Keep the existing marker filtering rule so categories with fewer than 2 cells are skipped.

**Step 4: Commit**

```bash
git add experiments/analysis.py benchmark_analysis.py benchmark_runner.py README.md
git commit -m "feat: add annotation analysis outputs"
```

---

## Task 5: Rework the benchmark runner around the new leaderboard task

**Objective:** make the runner execute reference/query annotation runs, seeds, and baselines in one command.

**Files:**
- Modify: `benchmark_runner.py`
- Modify: `configs/benchmark.yaml`
- Modify: `scripts/run_benchmark.sh`
- Modify: `docs/benchmark.md`

**Step 1: Add runner modes**

Supported modes should include:
- `smoke` — PBMC quick sanity test
- `annotation` — main cell type annotation benchmark

**Step 2: Add the default annotation configuration**

Default benchmark command should look like:
```bash
bash scripts/run_benchmark.sh --mode annotation --seeds 0 1 2
```

**Step 3: Verify output layout**

Expected directory structure:
```text
runs/
  annotation/
    gutatlas_transfer/
      scgpt/seed_0/
      scgpt/seed_1/
      celltypist_builtin/seed_0/
      ...
      summary.json
      summary_agg.csv
```

**Step 4: Commit**

```bash
git add benchmark_runner.py configs/benchmark.yaml scripts/run_benchmark.sh docs/benchmark.md
git commit -m "feat: make benchmark runner annotation-first"
```

---

## Task 6: Smoke test, run the benchmark, and prepare the release

**Objective:** verify the new benchmark path end-to-end and make the repository ready for leaderboard-style iteration.

**Files:**
- All modified benchmark files
- `README.md`
- optional release files: `CITATION.cff`, `CONTRIBUTING.md`

**Step 1: Run smoke tests**

Run:
```bash
python -m py_compile real_pbmc68k_finetune.py benchmark_runner.py benchmark_baselines.py benchmark_analysis.py experiments/*.py
bash scripts/run_benchmark.sh --mode smoke --seeds 0
```
Expected: pass without traceback.

**Step 2: Run the annotation benchmark**

Run the main benchmark with multiple seeds.

Expected outputs:
- per-run checkpoints and logs
- summary tables by method
- analysis figures and CSVs
- a top-line comparison in the README

**Step 3: Lock the release**

Update README to clearly state:
- the leaderboard task
- the reference/query datasets
- the evaluation protocol
- the main metrics
- the best-performing checkpoint

**Step 4: Commit and push**

```bash
git add .
git commit -m "feat: add cell type annotation benchmark"
git push origin main
```

---

## Success criteria

The benchmark is ready to chase a leaderboard if all of the following are true:

- the main task is a real cell type annotation / label-transfer benchmark
- reference and query datasets have a documented shared label protocol
- scGPT is compared against a non-trivial annotation baseline such as CellTypist
- the runner can produce seed-aggregated summary tables
- the analysis directory contains plots and per-class tables
- the README states the exact protocol and results
- the repository remains reproducible from a fresh clone

## Notes

- Keep the old PBMC benchmark as a smoke test, not as the main leaderboard target.
- If the gut-atlas label mapping turns out messy, the first implementation task should be to lock the label harmonization policy before any model tuning.
- Do not spend time on exotic architecture changes until the data protocol is fixed.
