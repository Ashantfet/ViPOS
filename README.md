# ViPOS — Visual Indoor Positioning System (MTP Project)

**ViPOS** (Visual Indoor Positioning System) is a DGX-trained hybrid deep learning framework that estimates the **camera's 6DoF pose relative to a fiducial tag** from a single RGB image. ViPOS focuses on the inverse pose problem: **predict camera pose w.r.t. tag**, using large synthetic datasets and a hybrid coarse-bin + fine-regression model.

---

## Table of Contents

* Overview
* Key Contributions (so far)
* Datasets
* Experiments & Results (summary)
* Repo structure
* Quick start (run / reproduce)
* Metrics and evaluation
* Tasks — Completed ✅ / To do ⏳
* Roadmap to publication (CVPR-ready)
* Deployment & productization
* Contact & acknowledgements

---

## Overview

ViPOS is designed to estimate the **camera's translation and orientation (x, y, z, roll, pitch, yaw)** in the coordinate frame of a visual tag (e.g., AprilTag / custom fiducial). The system is trained on large-scale synthetic data rendered in Blender and uses a hybrid model (CNN + Transformer) with a 2-stage coarse-bin classification (100 bins) followed by fine regression for high-precision pose prediction.

This inverse formulation (camera w.r.t tag) is useful for indoor localization, AR alignment, and multi-camera calibration where camera-centric coordinates simplify downstream tasks.

---

## Key Contributions (so far)

* **Inverse-pose formulation**: Directly predicting camera pose relative to a tag (camera→tag) instead of the widely-used tag→camera direction.
* **Large synthetic dataset (triangle → apriltag)**: Initially 100k triangle-based images (normalized 5DoF labels), later pivoted to 10k Apriltag-based images with full 6DoF labels for real-world transfer.
* **Hybrid coarse-bin + regression architecture**: Empirical finding that a hybrid model with **100 bins** offered the best generalization among tested bin sizes.
* **Hybrid CNN + Transformer backbone**: Demonstrated better performance than pure ViT or CNN baselines on your datasets.

---

## Datasets

* `synthetic/scalene_triangle_100k/` — 100k Blender-rendered images (earlier experiments; normalized 5DoF). Metadata includes normalized camera translations and angles.
* `synthetic/apriltag_10k/` — 10k Blender-rendered images with a single AprilTag and full 6DoF camera pose labels (ground-truth in tag frame).
* `real/` (planned) — small-scale real capture for sim2real validation (200–1000 images recommended).

---

## Experiments & Results (summary)

* Tested multiple architectures (ResNet variants, ViT, hybrid CNN+Transformer). The hybrid model with **100 bins** generalized best on triangle dataset experiments (detailed tables are in `docs/Training done till now (3).pdf`).
* Transition to Apriltag dataset improved real-world inference capability though the learning task complexity increased (6DoF labeling).

---

## Repo structure (suggested)

```
MTP_project/
├── data/
│   ├── synthetic/
│   │   ├── scalene_triangle_100k/
│   │   └── apriltag_10k/
│   └── real/  # placeholder for real captures
├── scripts/  # Blender scripts & metadata exporters
├── models/        # model definitions
├── train/         # training scripts
├── eval/          # evaluation scripts & notebooks
├── docs/          # reports, figures, pdfs
├── demos/         # inference scripts, demo server
├── configs/       # experiment configs
├── main.py
├── requirements.txt
└── README.md
```

---

## Quick start (reproduce top experiment)

1. Create & activate env:

```bash
conda create -n vipos python=3.10 -y
conda activate vipos
pip install -r requirements.txt
```

2. Place dataset under `data/synthetic/apriltag_10k/`.
3. Train (example config for hybrid_100bins):

```bash
python main.py --config configs/hybrid_100bins.yaml --mode train
```

4. Evaluate:

```bash
python main.py --config configs/hybrid_100bins.yaml --mode eval --checkpoint weights/best_hybrid_100bins.pth
```

Add `--device cuda` if running on DGX; adjust batch sizes per GPU memory.

---

## Metrics & evaluation protocol

* **Position error (cm)** — Mean / Median L2 between predicted & GT translation.
* **Orientation error (deg)** — Geodesic rotation distance on SO(3).
* **Bin accuracy** — if using bin classification: top-1 and top-3.
* **Robustness tests** — partial occlusion, lighting changes, intrinsics mismatch.
* **Sim2Real gap** — plot per-sample translation/rotation errors for synthetic vs real.

---

## Tasks — Completed ✅ / To do ⏳

### Completed ✅

* [x] Generated 100k triangle-based synthetic dataset (normalized 5DoF labels).
* [x] Trained multiple architectures and compiled training report (`docs/Training done till now (3).pdf`).
* [x] Identified hybrid CNN+Transformer and 100-bin formulation as best generalizing setup on triangle data.
* [x] Surveyed fiducial/tag literature and asserted the gap: most work outputs tag→camera; our goal is camera→tag.
* [x] Generated Apriltag-based synthetic dataset (10k images) with full 6DoF camera labels.

### To do ⏳

* [ ] Collect a small real dataset (200–1000 frames) with ground-truth poses for sim2real evaluation.
* [ ] Implement ROI-crop pipeline (tag crop + optional tag-6DoF auxiliary input) and run experiments.
* [ ] Add tag-identity augmentation and test multi-tag disambiguation.
* [ ] Implement uncertainty head and evaluate uncertainty calibration.
* [ ] Complete ablation studies for CVPR submission (bin sizes, backbones, ROI vs full image, aux inputs).
* [ ] Optimize, quantize, and export a mobile-friendly model (ONNX/TFLite) and create a demo.
* [ ] Prepare figures, tables, and reproducibility checklist for paper submission.

---

## Roadmap to CVPR-ready submission

1. Extend sim2real tests: capture real images and fine-tune on small real set.
2. Benchmark against classical PnP (openCV solvePnP on detected tag corners) and DeepTag/AprilTag baselines (convert their outputs to camera→tag). Report metric tables & runtime.
3. Run comprehensive ablations and provide error CDFs, heatmaps, and failure-case visualizations.
4. Add uncertainty modeling and fusion rules with PnP for low-confidence cases.
5. Prepare code & data release plan; include Dockerfile and seed-controlled configs.

---

## Deployment & productization checklist

* Export model to ONNX and run inference latency tests on target devices.
* Convert to INT8 where possible and verify accuracy drop.
* Create a lightweight demo app (mobile/web) with live camera feed + overlay of pose.
* Provide SDK + API for pose query: `POST /predict` → image → JSON pose + confidence.

---

## Contact & acknowledgements

Project owner: **[ASHANT KUMAR/Handle]**

If you want, I can:

* Export this README.md to the repo root and commit it for you (provide repo path or remote),
* Generate `configs/hybrid_100bins.yaml` and a `train.sh` helper script,
* Prepare the sim2real capture protocol and small annotation tool.

---

*End of README.*