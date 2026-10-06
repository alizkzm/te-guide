# TE-Bench

TE-Bench is a benchmark for **transferability estimation (TE)**. Given a hub of pre-trained models
and a target task, a TE score tries to rank the models by how well they will perform after
fine-tuning on the target, without fine-tuning any of them. TE-Bench runs published TE scores and
a set of simple baselines under one shared protocol, and compares their rankings with published
fine-tuning results.

The code lives in the `guide` package and is driven by a single command-line entry point, `run.py`.

## Installation

```bash
pip install -r requirements.txt
```

The requirements are NumPy, SciPy, scikit-learn, PyTorch, torchvision, timm, HuggingFace
`datasets` and pandas.

## The pipeline

A full run has four steps. Each step reads the output of the previous one and caches its own, so
you only pay for the expensive part once.

```
extract  ->  score  ->  post  ->  evaluate
 (GPU)       (CPU)      (fast)    (fast)
```

### 1. `extract`: one forward pass per model and target

Every model is run once over every target dataset. Its penultimate features, the logits of its
source classifier (when it has one) and the target labels are stored as a *probe*. This is the
only step that needs a GPU, and every score afterwards reads only these probes.

```bash
python run.py extract -m cnn -d all --device cuda
```

By default the same pass is also run on 5,000 ImageNet validation images (the *source probe*),
which a few scores such as OTCE, ATC and PED use. Pass `--no-source` to skip it.

### 2. `score`: compute the scores

```bash
python run.py score -m cnn -s all
```

Each (target, model, score) gives one small JSON record. `-s` takes `all`, `te`, `trivial`,
`complete` or a comma-separated list of names. Two options are useful for experiments:
`--samples-per-class 10` keeps a stratified subset of the target (seed 42 by default), and
`--set score.param=value` overrides a hyper-parameter.

### 3. `post`: combine and normalise

```bash
python run.py post -m cnn
```

Some scores have several terms (for example ETran, NCTI, PED and SA). This step combines them with
the recipe given by their authors and normalises every score over the models of the hub.

### 4. `evaluate`: compare with the ground truth

```bash
python run.py evaluate --gt sfda_cnn
```

For every target, each score's ranking of the models is compared with the fine-tuning accuracies,
and the results are averaged over the targets. `--detail lda` prints the per-target table for one
score, and `--top 10` keeps the ten best.

The four steps can also be run in one go:

```bash
python run.py all -m cnn --device cuda --gt sfda_cnn
```

`python run.py list` prints every available score, whether it is a published TE method or a
baseline, and whether it needs target labels.

## Coverage

### Model hubs and ground truths

| Hub (`-m`) | Models | Ground truth (`--gt`) |
|---|---|---|
| `cnn` | 11 supervised CNNs (ResNet, DenseNet, GoogLeNet, Inception-v3, MnasNet, MobileNet-v2) | `sfda_cnn` |
| `cnn_ssl` | self-supervised ResNet-50 checkpoints (MoCo, SimCLR, BYOL, SwAV, DeepCluster, ...) | `sfda_cnn_ssl` |
| `vit` | 10 vision transformers (ViT, PVT, Swin, DINO, MoCo-v3) | `sfda_vit` |
| `itm` | 20 mixed CNNs and transformers, supervised and self-supervised | `itm` |
| `all` | every model above | |

The ground truths are published fine-tuning accuracies: the SFDA tables for the three architecture
hubs and the ITM table for the mixed hub. `--gt` also accepts a path to your own table, either a
JSON file `{dataset: {model: accuracy}}` or a CSV with `dataset,model,accuracy` columns. The mixed
hub was fine-tuned from torchvision's newer weights, so extract it with `GUIDE_WEIGHTS=default`.

### Target datasets

Aircraft, Caltech-101, Cars, CIFAR-10, CIFAR-100, DTD, Flowers, Food-101, Pets, SUN397 and
VOC2007. On the command line they are `aircraft, caltech101, cars, cifar10, cifar100, dtd,
flowers, food, pets, sun397, voc2007`; `-d all` selects all of them.

### Scores

Published TE methods (separability, likelihood, evidence, transport and adaptation scores) and
baselines (capacity, prediction uncertainty, feature spectrum and weight statistics). Two scores,
NLEEP and PACTran, are disabled by default: `-s all` skips them, and they run when named
explicitly.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `GUIDE_ROOT` | `runs/` | where probes, scores and results are written |
| `GUIDE_DATA` | `data/` | where the target datasets are read from |
| `GUIDE_PROBES` | `$GUIDE_ROOT/probes` | probe cache, can be shared between runs |
| `GUIDE_WEIGHTS` | `v1` | torchvision weights (`v1` or `default`) |
| `GUIDE_IMAGENET_DIR` | | ImageNet folder for the source probe |

## Output

```
runs/
  probes/<source>/<model>/<target>.npz          features, logits and labels
  scores/<source>/<target>_<model>_<score>.json one record per score
  postprocessed/<source>/<target>_<score>.json  combined, normalised scores over the hub
  evaluation/<source>/
    summary_<gt>.csv                            one row per score, averaged over targets
    per_dataset_<gt>.csv                        one row per (score, target)
    evaluation_<gt>.json                        both, in one file
```

A score record keeps the raw values together with everything needed to reproduce them:

```json
{
  "target_data": "cifar100",
  "model_name": "resnet50",
  "score_name": "lda",
  "samples_per_class": "all",
  "score_elements_list": {"lda": 0.9853},
  "score_kind": "TE",
  "label_free": false,
  "n_samples": 60000,
  "n_classes": 100,
  "feature_dim": 2048,
  "hyperparameters": {},
  "seed": 42,
  "run_time": 18.4,
  "status": "ok"
}
```

`evaluate` prints a summary sorted by the weighted Kendall correlation:

```
ground truth: sfda_cnn   hub: 11 models   datasets: 11

score              kind     cmp         LF  weighted_kendall   pearson   kendall   rel@1   rec@3      sec
----------------------------------------------------------------------------------------------------------
w_alpha_weighted   TE       complete    y             0.7210    0.9077    0.7766  0.9952   0.909      0.0
itm                TE       complete    n             0.6745    0.6280    0.6153  0.9951   0.909    192.0
ncti               TE       complete    n             0.6694    0.6108    0.6011  0.9944   0.818    242.4
sfda               TE       complete    n             0.6592    0.5649    0.5712  0.9935   0.909    454.8
inb                TRIVIAL  complete    y             0.6543    0.5979    0.5680  0.9991   0.909      0.0
...
```

| Column | Meaning |
|---|---|
| `kind` | `TE` for a published method, `TRIVIAL` for a baseline |
| `cmp` | `complete`, or `incomplete` when the score needs an artefact the probes do not contain |
| `LF` | `y` if the score uses no target labels |
| `weighted_kendall` | weighted Kendall correlation with the ground truth, averaged over targets |
| `pearson`, `kendall` | linear and rank correlation, averaged over targets |
| `rel@1` | accuracy of the top-ranked model relative to the best model |
| `rec@3` | how often the top-ranked model is among the three best |
| `sec` | total run time of the score over the hub, in seconds |

The CSV files additionally report the Spearman correlation, the weighted Pearson correlation,
`rel@3`, `recall@1` and the top-1 regret.

## Testing without models or data

`selftest` runs every score on synthetic data, and `mock` writes synthetic probes so that the
whole pipeline can be exercised without downloading anything:

```bash
python run.py selftest --fast
GUIDE_ROOT=runs_mock python run.py mock -m cnn -d cifar10,dtd
GUIDE_ROOT=runs_mock python run.py score -m cnn -d cifar10,dtd
GUIDE_ROOT=runs_mock python run.py post -d cifar10,dtd
GUIDE_ROOT=runs_mock python run.py evaluate --gt sfda_cnn -d cifar10,dtd
```
