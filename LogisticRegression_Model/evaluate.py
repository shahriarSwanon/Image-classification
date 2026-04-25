"""
evaluate.py
-----------
Evaluates the trained model and shows a full graphical results dashboard:

  Panel 1  - Confusion Matrix heatmap
  Panel 2  - Per-class Accuracy bar chart
  Panel 3  - Precision / Recall / F1 grouped bar chart
  Panel 4  - Summary scorecard (overall accuracy + pass/fail)

All plots are saved to results/ AND displayed on screen.

Usage:
    python evaluate.py          # standalone
    from evaluate import evaluate  # called from main.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config

# Use a non-interactive backend only when no display is available
# (keeps plt.show() working on Windows)
matplotlib.rcParams["figure.dpi"] = 110
plt.rcParams["font.family"] = "DejaVu Sans"

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0f1117",
    "panel":     "#1a1d27",
    "accent1":   "#4f8ef7",   # blue
    "accent2":   "#f7914f",   # orange
    "accent3":   "#4fc98e",   # green
    "accent4":   "#f74f6e",   # red
    "text":      "#e8eaf0",
    "subtext":   "#8b8fa8",
    "grid":      "#2a2d3a",
    "success":   "#4fc98e",
    "fail":      "#f74f6e",
}

CLASS_COLORS = ["#4f8ef7", "#f7914f", "#4fc98e", "#f74fca", "#f7e24f"]


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate(
    clf,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    save_dir: str = config.RESULTS_DIR,
    show: bool    = True,
) -> dict:
    """
    Runs predictions, prints the report, shows the dashboard, saves PNGs.

    Returns
    -------
    metrics : dict with keys 'accuracy', 'report', 'confusion_matrix'
    """
    X_scaled = scaler.transform(X_test) if scaler is not None else X_test
    y_pred   = clf.predict(X_scaled)

    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=class_names)
    cm      = confusion_matrix(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(class_names)))
    )

    _print_banner(acc)
    print(report)

    dashboard_path = _build_dashboard(
        cm, class_names, acc, prec, rec, f1, save_dir, show
    )
    print(f"[Evaluate] Dashboard saved -> {dashboard_path}")

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_banner(acc: float) -> None:
    border = "=" * 50
    tag    = "TARGET MET (>=90%)" if acc >= 0.90 else "Below 90% target"
    mark   = "[OK]" if acc >= 0.90 else "[!!]"
    print(f"\n{border}")
    print(f"  FINAL TEST ACCURACY : {acc:.2%}   {mark} {tag}")
    print(f"{border}\n")


# ── Dashboard builder ─────────────────────────────────────────────────────────

def _build_dashboard(
    cm:          np.ndarray,
    class_names: list,
    acc:         float,
    prec:        np.ndarray,
    rec:         np.ndarray,
    f1:          np.ndarray,
    save_dir:    str,
    show:        bool,
) -> Path:
    """Builds and optionally shows a 4-panel results dashboard."""

    fig = plt.figure(figsize=(18, 13), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Animal Classifier  |  Logistic Regression + MobileNetV2",
        fontsize=18, fontweight="bold",
        color=PALETTE["text"], y=0.97,
    )

    # Grid: 2 rows x 3 cols; panels span as noted
    gs = fig.add_gridspec(
        2, 3,
        hspace=0.38, wspace=0.35,
        left=0.06, right=0.97, top=0.92, bottom=0.07,
    )

    ax_cm      = fig.add_subplot(gs[0, 0:2])   # confusion matrix (wide)
    ax_summary = fig.add_subplot(gs[0, 2])     # scorecard
    ax_bar     = fig.add_subplot(gs[1, 0])     # per-class accuracy
    ax_prf     = fig.add_subplot(gs[1, 1:3])   # precision / recall / f1

    _panel_confusion(ax_cm,  cm, class_names, acc)
    _panel_summary(ax_summary, acc, prec, rec, f1, class_names)
    _panel_per_class(ax_bar, cm, class_names)
    _panel_prf(ax_prf, prec, rec, f1, class_names)

    out = Path(save_dir) / "results_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])

    if show:
        print("[Evaluate] Displaying interactive dashboard (close window to exit)...")
        plt.show()
    else:
        plt.close(fig)

    return out


# ── Panel 1: Confusion Matrix ─────────────────────────────────────────────────

def _panel_confusion(ax, cm, class_names, acc):
    _style_ax(ax)

    # Normalised % matrix for colour mapping, raw counts as annotations
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annot = np.array([
        [f"{cm[i,j]}\n({cm_norm[i,j]:.0%})" for j in range(len(class_names))]
        for i in range(len(class_names))
    ])

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.6,
        linecolor=PALETTE["grid"],
        cbar_kws={"shrink": 0.85},
        vmin=0, vmax=1,
    )

    ax.set_title(
        f"Confusion Matrix  (Overall Accuracy: {acc:.2%})",
        color=PALETTE["text"], fontsize=13, pad=10,
    )
    ax.set_xlabel("Predicted", color=PALETTE["subtext"], fontsize=11)
    ax.set_ylabel("Actual",    color=PALETTE["subtext"], fontsize=11)
    ax.tick_params(colors=PALETTE["text"])
    ax.figure.axes[-1].tick_params(colors=PALETTE["text"])   # colorbar ticks


# ── Panel 2: Summary Scorecard ────────────────────────────────────────────────

def _panel_summary(ax, acc, prec, rec, f1, class_names):
    _style_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background card
    card = mpatches.FancyBboxPatch(
        (0.05, 0.03), 0.90, 0.94,
        boxstyle="round,pad=0.03",
        facecolor=PALETTE["panel"],
        edgecolor=PALETTE["accent1"],
        linewidth=1.5,
    )
    ax.add_patch(card)

    # Title
    ax.text(0.5, 0.91, "Results Summary", ha="center", va="center",
            fontsize=13, fontweight="bold", color=PALETTE["text"])

    # Big accuracy number
    color = PALETTE["success"] if acc >= 0.90 else PALETTE["fail"]
    ax.text(0.5, 0.73, f"{acc:.1%}", ha="center", va="center",
            fontsize=36, fontweight="bold", color=color)

    status = "TARGET MET" if acc >= 0.90 else "BELOW TARGET"
    ax.text(0.5, 0.60, status, ha="center", va="center",
            fontsize=11, fontweight="bold", color=color)

    ax.axhline(y=0.55, xmin=0.1, xmax=0.9,
               color=PALETTE["grid"], linewidth=1)

    # Macro averages
    rows = [
        ("Macro Precision", f"{prec.mean():.3f}"),
        ("Macro Recall",    f"{rec.mean():.3f}"),
        ("Macro F1-Score",  f"{f1.mean():.3f}"),
        ("# Classes",       str(len(class_names))),
    ]
    for i, (label, val) in enumerate(rows):
        y = 0.47 - i * 0.10
        ax.text(0.15, y, label, ha="left",  va="center",
                fontsize=9, color=PALETTE["subtext"])
        ax.text(0.85, y, val,   ha="right", va="center",
                fontsize=10, fontweight="bold", color=PALETTE["text"])

    ax.set_title("Scorecard", color=PALETTE["text"], fontsize=13, pad=10)


# ── Panel 3: Per-Class Accuracy ───────────────────────────────────────────────

def _panel_per_class(ax, cm, class_names):
    _style_ax(ax)

    per_class = cm.diagonal() / cm.sum(axis=1)
    colors = [
        PALETTE["success"] if v >= 0.90 else PALETTE["fail"]
        for v in per_class
    ]

    bars = ax.barh(class_names, per_class * 100, color=colors,
                   edgecolor="none", height=0.55)

    for bar, val in zip(bars, per_class):
        ax.text(
            min(bar.get_width() + 1.5, 98), bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left",
            fontsize=9, fontweight="bold", color=PALETTE["text"],
        )

    ax.axvline(90, color=PALETTE["accent4"], linestyle="--",
               linewidth=1.3, label="90% target")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Accuracy (%)", color=PALETTE["subtext"], fontsize=10)
    ax.set_title("Per-Class Accuracy", color=PALETTE["text"], fontsize=13, pad=10)
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8)

    ok_patch   = mpatches.Patch(color=PALETTE["success"], label=">= 90%")
    fail_patch = mpatches.Patch(color=PALETTE["fail"],    label="< 90%")
    ax.legend(handles=[ok_patch, fail_patch],
              facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8, loc="lower right")


# ── Panel 4: Precision / Recall / F1 ─────────────────────────────────────────

def _panel_prf(ax, prec, rec, f1, class_names):
    _style_ax(ax)

    n      = len(class_names)
    x      = np.arange(n)
    width  = 0.25

    b1 = ax.bar(x - width, prec * 100, width, label="Precision",
                color=PALETTE["accent1"], edgecolor="none")
    b2 = ax.bar(x,          rec  * 100, width, label="Recall",
                color=PALETTE["accent2"], edgecolor="none")
    b3 = ax.bar(x + width,  f1   * 100, width, label="F1-Score",
                color=PALETTE["accent3"], edgecolor="none")

    for bars in (b1, b2, b3):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{bar.get_height():.0f}",
                ha="center", va="bottom",
                fontsize=7.5, color=PALETTE["text"],
            )

    ax.axhline(90, color=PALETTE["accent4"], linestyle="--",
               linewidth=1.2, label="90% target")
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, color=PALETTE["text"], fontsize=10)
    ax.set_ylabel("Score (%)", color=PALETTE["subtext"], fontsize=10)
    ax.set_title("Precision / Recall / F1 per Class",
                 color=PALETTE["text"], fontsize=13, pad=10)
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=9)


# ── Style helper ──────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["subtext"])
    ax.yaxis.label.set_color(PALETTE["subtext"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, alpha=0.7)


# ── Standalone Run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    from sklearn.model_selection import train_test_split

    from preprocessing import get_dataloader, validate_dataset
    from feature_extraction import extract_features

    print("=" * 50)
    print("  EVALUATION  (standalone run)")
    print("=" * 50)

    if not Path(config.MODEL_SAVE_PATH).exists():
        print(f"[Evaluate] Model not found at '{config.MODEL_SAVE_PATH}'.")
        print("[Evaluate] Please run  python train.py  first.")
        raise SystemExit(1)

    with open(config.MODEL_SAVE_PATH, "rb") as f:
        payload = pickle.load(f)
    clf, scaler = payload["model"], payload["scaler"]

    dataset_dir = (
        config.AUGMENTED_DIR
        if Path(config.AUGMENTED_DIR).exists()
        else config.DATASET_DIR
    )
    if not validate_dataset(dataset_dir):
        raise SystemExit(1)

    loader, class_names = get_dataloader(dataset_dir)
    X, y                = extract_features(loader, class_names)

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )

    evaluate(clf, scaler, X_test, y_test, class_names, show=True)
