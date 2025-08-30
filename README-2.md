# CanNet‑VCD: A Lightweight Hybrid SAE–SVM for Visible Canine Disease Identification

> **TL;DR**: One **supervised autoencoder (SAE)** per disease class learns a compact **128‑D latent**; a **calibrated RBF‑SVM** performs 4‑way classification on latents. At inference, **only one encoder** (~6.446M params) + SVM runs, costing **~0.083 GFLOPs** per 224×224 image—**edge‑friendly** with **≈0.99 accuracy / macro‑F1** on our test set.

## Highlights
- **Low compute**: ~0.083 GFLOPs / image (224×224) for encoder + SVM head.  
- **High accuracy**: ≈0.99 accuracy and macro‑F1 on four visible diseases.  
- **Modular & scalable**: add a disease by training **one** SAE; **refit SVM** (no end‑to‑end retrain).  
- **Explainable**: Score‑CAM overlays + t‑SNE/UMAP of latents; probability calibration (reliability diagrams, Brier score).  
- **Reproducible**: fixed splits, seeds, and a path to reproduce **without raw images** using shared latents.

---

## Repository structure
```
cannet-vcd/
  README.md
  REPRODUCIBILITY.md
  LICENSE                 # Code license (e.g., Apache-2.0)
  CITATION.cff
  requirements.txt        # or environment.yml
  configs/
    config_224.yaml       # training hyperparameters (latent_dim=128, etc.)
    classes_4.yaml        # class names and indices
  splits/
    split_v1_train.txt
    split_v1_val.txt
    split_v1_test.txt     # image paths (relative), deterministic seed
  src/
    models/sae_224.py
    train/train_sae_per_class.py
    train/extract_latents.py
    eval/svm_train_eval.py
    eval/mlp_train_eval.py
    eval/roc_curves.py
    eval/confusion_matrix.py
    eval/calibration_reliability.py
    eval/metrics_stats.py      # Wilson/bootstrapped CI, Cohen’s h, Brier/ECE
    explain/scorecam.py
    utils/seed_utils.py
    utils/flops_utils.py
  artifacts/
    svm.pkl                   # StandardScaler + RBF-SVM + Platt calibration
    latents_224_v1.npz        # Optional: train/val/test latents (128-D) + labels
  weights/
    encoder_cpv.pth
    encoder_distemper.pth
    encoder_mammary.pth
    encoder_mange.pth
  figures/
    confusion_matrix.png
    roc_curves.png
    reliability_diagram.png
    tsne_latent.png
    umap_latent.png
  data/
    samples/                  # small de-identified examples for quick tests
    DATASET.md                # how to obtain the full dataset
```

> If files exceed GitHub’s 100 MB limit, track with **Git LFS**, attach to a **Release**, or mirror to **Zenodo/HuggingFace**.

---

## Installation

```bash
# 1) Create and activate a fresh environment (example: Python 3.10–3.12)
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Apple Silicon) Verify Metal backend
python - << 'PY'
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
PY
```

**CUDA users**: ensure a torch build matching your CUDA toolkit; see pytorch.org.

---

## Data

- The **full dataset** is not redistributed. See `data/DATASET.md` for instructions to obtain images and layout folders as:
  ```
  CanineNIBLD23/
    Canine_distemper/
    Canine_Parvo_Virus/
    Mammary_Tumor/
    Mange/
  ```
- To ensure exact reproducibility, we provide fixed file lists in `splits/`.  
- For a **no‑data** path, use the shared **latents** (`artifacts/latents_224_v1.npz`) to reproduce all classifier metrics and plots.

---

## Quick start (no raw images required)

```bash
# Use provided latents to train & evaluate SVM and make figures
python -m src.eval.svm_train_eval        --latents artifacts/latents_224_v1.npz --out artifacts/svm.pkl
python -m src.eval.confusion_matrix      --latents artifacts/latents_224_v1.npz --model artifacts/svm.pkl
python -m src.eval.roc_curves            --latents artifacts/latents_224_v1.npz --model artifacts/svm.pkl
python -m src.eval.calibration_reliability --latents artifacts/latents_224_v1.npz --model artifacts/svm.pkl
python -m src.eval.metrics_stats         --latents artifacts/latents_224_v1.npz --model artifacts/svm.pkl
```

Expected: accuracy and macro‑F1 ≈ 0.99; macro/micro AUC ≳ 0.99; clean confusion matrix and near‑diagonal calibration.

---

## Full pipeline (with raw images)

### 1) Train one SAE **per class** (224×224)
```bash
python -m src.train.train_sae_per_class --config configs/config_224.yaml \
  --root CanineNIBLD23 --splits splits/split_v1_train.txt --val splits/split_v1_val.txt \
  --outdir weights/
```
- Architecture: 3× stride‑2 conv blocks → Flatten(50,176) → FC → **128‑D latent**.  
- Loss: **MSE reconstruction**; decoder used **only** during training.

### 2) Extract **128‑D latents**
```bash
python -m src.train.extract_latents --root CanineNIBLD23 \
  --splits splits/split_v1_*.txt --weights weights/ --out artifacts/latents_224_v1.npz
```

### 3) Train + evaluate **RBF‑SVM** (with Platt calibration)
```bash
python -m src.eval.svm_train_eval --latents artifacts/latents_224_v1.npz --out artifacts/svm.pkl
```

### 4) Plots and statistics
```bash
python -m src.eval.confusion_matrix          --latents artifacts/latents_224_v1.npz --model artifacts/svm.pkl
python -m src.eval.roc_curves                --latents artifacts/latents_224_v1.npz --model artifacts/svm.pkl
python -m src.eval.calibration_reliability   --latents artifacts/latents_224_v1.npz --model artifacts/svm.pkl
python -m src.eval.metrics_stats             --latents artifacts/latents_224_v1.npz --model artifacts/svm.pkl
```
- Reports: **95% CIs** (Wilson/bootstrapped), **Cohen’s h** effect sizes, **Brier score** & **ECE** for calibration.

---

## Explainability

- **Score‑CAM** on the final encoder block (gradient‑free; clearer contiguous heatmaps):
  ```bash
  python -m src.explain.scorecam --weights weights/encoder_distemper.pth \
      --image data/samples/distemper_001.jpg --class-name Canine_distemper --out figures/scorecam.png
  ```
- **Latent visualizations**:  
  ```bash
  python -m src.eval.plot_tsne_umap --latents artifacts/latents_224_v1.npz --out figures/
  ```

---

## Accuracy vs. efficiency

- **Compute**: ~**0.083 GFLOPs** / image (encoder + SVM; **1 MAC = 2 FLOPs**).  
- **Parameters (inference)**: ~**6.446M** (encoder only).  
- **Scalability**: add a disease by training **one** new SAE; **refit SVM** on the expanded latent bank.

A utility script `utils/flops_utils.py` reproduces FLOP estimates.

---

## Reproducibility notes
```python
# src/utils/seed_utils.py (used in all scripts)
import numpy as np, random, torch
def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(42); torch.manual_seed(42)
    if torch.backends.mps.is_available(): pass  # deterministic ops limited on MPS
    torch.use_deterministic_algorithms(False)
```
- We fix data splits and seeds; minor variance (±0.1%) may occur on MPS.

---

## License

- **Code**: Apache-2.0 (see `LICENSE`).  
- **Weights / Latents (if provided)**: e.g., **CC BY‑NC 4.0** (see `MODEL_LICENSE`).  
- **Data**: not redistributed; follow original dataset terms (`data/DATASET.md`).

---

## Citation

If you use this code or results, please cite:

```bibtex
@misc{cannetvcd2025,
  title   = {CanNet-VCD: A Lightweight Hybrid SAE–SVM for Visible Canine Disease Identification},
  author  = {<Your Name> and <Coauthors>},
  year    = {2025},
  url     = {https://github.com/<your-org>/cannet-vcd},
  note    = {Code and reproducible artifacts}
}
```

A `CITATION.cff` is included so GitHub renders a “Cite this repository” box.

---

## Troubleshooting

- **MPS not available (Apple silicon)**: ensure recent PyTorch; try Python 3.10–3.12; otherwise use CPU fallback.  
- **CUDA OOM**: reduce batch size in `config_224.yaml`.  
- **Different results**: verify you used our `splits/` and the same `latent_dim=128`; confirm scaler+SVM pickle matches the latent file.  
- **Large files**: track with Git LFS and attach to a GitHub Release.

---

## Contact
Open an issue or email **<your.email@domain>** for questions/bug reports.
