#chandan.iasri 
#make_scorecam_figure.py
# Creates (1) triptych with caption using inferno colormap
# and (2) a grayscale heatmap variant. Saves PNG (600 dpi) + PDF.

# =============== USER SETTINGS ===============
ORIGINAL_PATH = "CanineNIBLD23/Canine_distemper/0a933a4b-3e61-4518-abec-4f41f662ce83.Jpg"#change accordingly
OVERLAY_PATH  = "GRAD_Output/_overlay.png"     # blended overlay you already have
HEATMAP_PATH  = "GRAD_Output/_heatmap.png"     # raw heatmap image (any RGB); or .npy array

OUT_PREFIX    = "GRAD_Output_images/"
CAPTION_TEXT  = (
    "Figure X. Score-CAM visualization for the predicted class. "
    "Warmer colors indicate pixels that most increase the classifier’s confidence. "
    "The model primarily attends to pad margins and the interdigital area, "
    "while background edges show limited activation."
)
DPI           = 600
# ============================================

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from textwrap import wrap

# Pillow resampling for compatibility
try:
    RESAMP = Image.Resampling.BILINEAR
except AttributeError:
    RESAMP = Image.BILINEAR

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_heatmap_array(hpath, target_size=None):
    """
    Loads a heatmap as a [0,1] float array (H,W).
    Accepts:
      - .npy file (H,W) or (H,W,1) or (H,W,3)
      - image file (RGB/gray)
    """
    if hpath.lower().endswith(".npy"):
        arr = np.load(hpath)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        arr = arr.astype(np.float32)
        # normalize if not already in [0,1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        if target_size is not None:
            arr_img = Image.fromarray((arr * 255).astype(np.uint8)).resize(target_size, RESAMP)
            arr = np.asarray(arr_img).astype(np.float32) / 255.0
        return arr
    else:
        im = Image.open(hpath).convert("RGB")
        if target_size is not None:
            im = im.resize(target_size, RESAMP)
        rgb = np.asarray(im).astype(np.float32) / 255.0
        gray = rgb.mean(axis=2)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        return gray

def add_caption(fig, caption, fontsize=9, pad=0.01):
    """Add a wrapped caption centered below the subplots."""
    wrapped = "\n".join(wrap(caption, width=120))
    fig.text(0.5, pad, wrapped, ha="center", va="bottom", fontsize=fontsize)

def make_triptych(original_path, overlay_path, heatmap_path, out_prefix,
                  caption_text, cmap="inferno", dpi=600, grayscale_variant=True):
    # Load images
    orig = Image.open(original_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB").resize(orig.size, RESAMP)
    H, W = orig.size[1], orig.size[0]

    heat = load_heatmap_array(heatmap_path, target_size=orig.size)  # (H,W) in [0,1]

    # ---------- (A) Inferno variant ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=dpi)
    for ax in axes: ax.axis("off")

    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=10)

    axes[1].imshow(overlay)
    axes[1].set_title("Score-CAM Overlay", fontsize=10)

    im = axes[2].imshow(heat, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[2].set_title("Score-CAM Heatmap", fontsize=10)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Class-evidence (0–1)", fontsize=8)

    plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave space for caption
    # add_caption(fig, caption_text, fontsize=9, pad=0.01)

    ensure_dir(out_prefix + "1.png")
    fig.savefig(out_prefix + "2.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_prefix + "3.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------- (B) Grayscale heatmap variant ----------
    if grayscale_variant:
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5), dpi=dpi)
        for ax in axes2: ax.axis("off")

        axes2[0].imshow(orig);        axes2[0].set_title("Original", fontsize=10)
        axes2[1].imshow(overlay);     axes2[1].set_title("Score-CAM Overlay", fontsize=10)
        im2 = axes2[2].imshow(heat, cmap="gray", vmin=0.0, vmax=1.0)
        axes2[2].set_title("Score-CAM Heatmap (grayscale)", fontsize=10)
        cbar2 = fig2.colorbar(im2, ax=axes2[2], fraction=0.046, pad=0.04)
        cbar2.set_label("Class-evidence (0–1)", fontsize=8)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        add_caption(fig2, caption_text, fontsize=9, pad=0.01)

        fig2.savefig(out_prefix + "1_gray.png", dpi=dpi, bbox_inches="tight")
        fig2.savefig(out_prefix + "2_gray.pdf", bbox_inches="tight")
        plt.close(fig2)

if __name__ == "__main__":
    make_triptych(
        ORIGINAL_PATH,
        OVERLAY_PATH,
        HEATMAP_PATH,
        OUT_PREFIX,
        CAPTION_TEXT,
        cmap="inferno",
        dpi=DPI,
        grayscale_variant=True
    )
    print("[OK] Saved:",
          OUT_PREFIX + ".png,",
          OUT_PREFIX + ".pdf,",
          OUT_PREFIX + "_gray.png,",
          OUT_PREFIX + "_gray.pdf")

