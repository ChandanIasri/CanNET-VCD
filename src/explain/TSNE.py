# === visualize_tsne_latent_4class.py ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ---------------- Config ----------------
latent_dir = "."                     # folder with latent_class_*.npy
data_root = "CanineNIBLD23"          # dataset root to get class names

# Classes to include (exclude 'Rickets')
include_classes = [
    "Canine_Parvo_Virus",
    "Canine_distemper",
    "Mammary_Tumor",
    "Mange"
]

class_names = [d for d in include_classes if os.path.isdir(os.path.join(data_root, d))]
num_classes = len(class_names)

# Check available class folders
print(f"Using classes: {class_names} (total={num_classes})")

# ------------- Load latents -------------
X_list, y_list = [], []
for i, cname in enumerate(class_names):
    latent_file = f"latent_class_{i}.npy"
    path = os.path.join(latent_dir, latent_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing latent file for {cname}: {path}")
    z = np.load(path)  # shape: (N_i, D)
    X_list.append(z)
    y_list.append(np.full(z.shape[0], i, dtype=int))

X = np.vstack(X_list)            # (N_total, D)
y = np.concatenate(y_list)       # (N_total,)
print(f"Loaded latents: X={X.shape}, per-class counts={np.bincount(y)}")

# --------- Standardize then t-SNE ----------
X_scaled = StandardScaler().fit_transform(X)

# Pick a safe perplexity (must be < number of points in the smallest class)
min_class_n = np.min(np.bincount(y))
perplexity = max(5, min(30, min_class_n // 2))  # cap at 30 by default
print(f"Using t-SNE perplexity={perplexity} (min class size={min_class_n})")

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate='auto',
    init='pca',
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled)  # (N, 2)

# --------- Save coordinates (CSV) ---------
df = pd.DataFrame({
    "tsne_1": X_tsne[:, 0],
    "tsne_2": X_tsne[:, 1],
    "label_id": y,
    "label_name": [class_names[i] for i in y]
})
df.to_csv("tsne_latent_coordinates_4class.csv", index=False)
print("Saved: tsne_latent_coordinates_4class.csv")

# --------------- Plot ----------------
plt.figure(figsize=(9, 7))
palette = sns.color_palette("Set2", n_colors=num_classes)

for i, name in enumerate(class_names):
    mask = (y == i)
    plt.scatter(
        X_tsne[mask, 0], X_tsne[mask, 1],
        s=8, alpha=0.75, label=name, color=palette[i]
    )
    # Annotate cluster centroid with class name
    if np.any(mask):
        cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
        plt.text(cx, cy, name, fontsize=9, weight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.6))

plt.title("t-SNE of SAE Latent Features (4 Diseases)", pad=10)
plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
plt.legend(markerscale=2, frameon=True)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("tsne_latent_4class.png", dpi=300)
plt.show()

print("Saved: tsne_latent_4class.png")
