from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def slice_report(data: np.ndarray, mask: np.ndarray, row: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    limits = np.percentile(data[mask], [1, 99])
    center = np.array(np.where(mask)).mean(axis=1).astype(int)
    for axis, ax, title in zip(range(3), axes, ["Sagittal", "Coronal", "Axial"]):
        pixels = np.take(data, center[axis], axis=axis).T
        overlay = np.take(mask, center[axis], axis=axis).T
        ax.imshow(pixels, origin="lower", cmap="gray", vmin=limits[0], vmax=limits[1])
        if overlay.any() and not overlay.all():
            ax.contour(overlay, levels=[0.5], colors=["#27c5b8"], linewidths=0.6)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{row['subject_id']} | {row['site']} | {row['status']} — foreground outline")
    fig.text(
        0.05,
        0.03,
        f"Foreground {row['foreground_volume_ml']:.1f} ml; entropy {row['entropy']:.2f} bits\n"
        "RAS-oriented; foreground includes non-brain tissue. See QC CSV for flags.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    save_figure(fig, path)
