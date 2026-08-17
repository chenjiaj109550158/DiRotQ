import numpy as np

from metrics.evaluate_shared_pca_audit import BASELINE, bootstrap_ci, screen_decision, summarize


def _rows(config, psnr, lpips, ssim, clip):
    return [
        {
            "image_id": str(index), "config": config,
            "psnr": p, "lpips": l, "ssim": s, "clip_score": c,
        }
        for index, (p, l, s, c) in enumerate(zip(psnr, lpips, ssim, clip))
    ]


def test_bootstrap_is_deterministic():
    values = np.asarray([1.0, 2.0, 3.0])
    assert bootstrap_ci(values, 100, 7) == bootstrap_ci(values, 100, 7)


def test_summary_direction_and_screen():
    baseline = _rows(BASELINE, [10, 11, 12], [.2, .2, .2], [.5] * 3, [20] * 3)
    candidate = _rows("shared", [10.2, 11.2, 12.2], [.19] * 3, [.51] * 3, [20.1] * 3)
    summary = summarize(baseline + candidate, bootstrap_samples=200)
    row = next(item for item in summary if item["config"] == "shared")
    assert row[f"psnr_delta_vs_{BASELINE}"] > 0
    assert row[f"lpips_delta_vs_{BASELINE}"] < 0
    assert row[f"psnr_win_rate_vs_{BASELINE}"] == 1
    assert row[f"lpips_win_rate_vs_{BASELINE}"] == 1
    assert screen_decision(summary)["shared"]["eligible_for_5k"]

