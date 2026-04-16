# PCM-NeRF: Probabilistic Camera Modeling for Neural Radiance Fields under Pose Uncertainty

<div align="center">
<img src="assets/CVPR2026.png" width="75%"/>
</div>

> **PCM-NeRF: Probabilistic Camera Modeling for Neural Radiance Fields under Pose Uncertainty.**  
> Shravan Venkatraman\*,  Rakesh Raj\*, Pavan Kumar S\*,
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops 2026  
> *(\* Equal Contribution)*

<!-- #### [project page](https://shravan-18.github.io/UGPL/) | [paper](https://arxiv.org/abs/2507.14102) -->

Probabilistic Camera Modeling for Neural Radiance Fields under Pose Uncertainty.

Built on top of [SG-NeRF](https://github.com/Gorilla-Lab-SCUT/SG-NeRF) and [NeuS](https://github.com/Totoro97/NeuS). Each camera pose is modeled as a distribution with a learnable mean and variance. Uncertainty regularization couples learned variance parameters to per-camera view confidence (derived from SfM correspondence quality and rolling PSNR), damping gradient updates for poorly-initialized views and preventing them from corrupting the reconstruction.

---

## Setup

```bash
pip install torch torchvision numpy trimesh open3d scipy pyhocon tqdm tensorboard
```

Feature matching (SuperPoint + SuperGlue) must be run separately before training to produce the correspondence files expected by `MatchingDataset`.

---

## Training

```bash
python exp_runner.py \
  --conf confs/teeth.conf \
  --case <scene_name> \
  --uncertainty_weight 0.05 \
  --iou_weight 0.2 \
  --iters 150000 \
  --batch_size 512 \
  --gpu 0
```

Resume from checkpoint:

```bash
python exp_runner.py --conf confs/teeth.conf --case <scene_name> -c
```

Resume from a specific checkpoint:

```bash
python exp_runner.py --conf confs/teeth.conf --case <scene_name> -c --ckpt ckpt_150000.pth
```

---

## Evaluation

Edit the `exp_dir` and `data_dir` paths at the bottom of `evaluate.py`, then:

```bash
python evaluate.py
```

Reports Chamfer Distance, Accuracy, Completeness, Precision, Recall, and F-score — computed both over all views and over pose inliers only (≤20 cm, ≤20°).

---

## Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--uncertainty_weight` | `0.05` | Weight λ_unc on the uncertainty regularisation loss |
| `--iou_weight` | `0.2` | Weight λ_IoU on the volumetric distribution alignment loss |
| `--iters` | `150000` | Total training iterations |
| `--batch_size` | `512` | Rays per iteration |
| `model.uncertainty.kappa` | `5.0` | Sensitivity scalar κ for per-camera gradient damping |
| `model.uncertainty.warmup_iters` | `10000` | Iterations before uncertainty loss activates |
| `model.uncertainty.rampup_iters` | `20000` | Iterations over which the loss ramps up to full weight |
