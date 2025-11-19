import numpy as np
from pathlib import Path

from admet.visualize.model_performance import plot_parity_grid, plot_metric_bars


def _make_synthetic():
    endpoints = ["LogD", "KSOL"]
    n = 20
    rng = np.random.default_rng(0)
    # create monotonic relationships with noise
    x = rng.normal(size=(n, 1))
    y = np.hstack([x[:, 0:1], (x * 0.5 + 1.0)])
    # predictions are noisy
    y_pred = y + rng.normal(scale=0.1, size=y.shape)
    mask = (~np.isnan(y)).astype(int)
    return endpoints, y, y_pred, mask


def test_plot_generation(tmp_path: Path):
    endpoints, y_true, y_pred, mask = _make_synthetic()
    y_true_dict = {"train": y_true, "val": y_true, "test": y_true}
    y_pred_dict = {"train": y_pred, "val": y_pred, "test": y_pred}
    mask_dict = {"train": mask, "val": mask, "test": mask}

    fig_root = tmp_path / "figures"
    for space in ["log", "linear"]:
        space_dir = fig_root / space
        plot_parity_grid(
            y_true_dict, y_pred_dict, mask_dict, endpoints, space=space, save_dir=space_dir, n_jobs=2
        )
        # check at least one file exists per endpoint
        for ep in endpoints:
            fname = space_dir / f"parity_{ep.replace(' ', '_').replace('/', '_')}.png"
            assert fname.exists() and fname.stat().st_size > 0

        # metric bars
        r2_path = space_dir / "metrics_r2.png"
        spr2_path = space_dir / "metrics_spearman_rho2.png"
        # metric bars for all supported metrics
        metric_paths = [
            space_dir / "metrics_mae.png",
            space_dir / "metrics_rmse.png",
            space_dir / "metrics_r2.png",
            space_dir / "metrics_pearson_r2.png",
            space_dir / "metrics_spearman_rho2.png",
            space_dir / "metrics_kendall_tau.png",
        ]
        plot_metric_bars(
            y_true_dict,
            y_pred_dict,
            mask_dict,
            endpoints,
            space=space,
            save_path_r2=r2_path,
            save_path_spr2=spr2_path,
            n_jobs=2,
        )
        for p in metric_paths:
            assert p.exists() and p.stat().st_size > 0
