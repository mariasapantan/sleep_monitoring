from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_rem_vs_nrem(y_true: np.ndarray, y_probs: np.ndarray) -> None:
    """
    Plot ROC curve for REM (5) vs NREM (1, 2, 3). Ignores wake (0).
    
    Args:
        y_true: Ground truth labels (0, 1, 2, 3, 5).
        y_probs: Predicted class probabilities, shape (n_samples, num_classes).
    """
    # Filter out wake (0) and keep only REM (5) and NREM (1, 2, 3)
    mask = np.isin(y_true, [1, 2, 3, 5])
    y_true_filtered = y_true[mask]
    y_scores_filtered = y_probs[mask, 5]  # probability for REM

    # Convert labels to binary: 1 for REM, 0 for NREM
    y_binary = (y_true_filtered == 5).astype(int)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_binary, y_scores_filtered)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: REM vs NREM")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_rem_vs_nrem.png", dpi=300, bbox_inches='tight')  # or .pdf
    plt.close()

def plot_loss_curve(losses: list) -> None:
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Training Loss')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()
