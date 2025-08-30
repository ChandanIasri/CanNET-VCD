# === visualize_umap_latent_4class.py ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap  # pip install umap-learn

# ---------------- Config ----------------
latent_dir = "."               # folder with latent_class_*.npy
data_root  = "CanineNIBLD23"   # dataset root (used only to sanity-check class folders)

# Include ONLY these 4 classes (Rickets excluded)
include_classes = [
    "Canine_Parvo_Virus",
    "Canine_distemper",
    "Mammary_Tumor",
    "Mange",
]

# If your saved files are not 0..3 for these classes, set a manual map:
# e.g., if Rickets was class_2 and the rest were 0,1,3,4 you can map like:
# manual_index_map = {
#     "Canine_Parvo_Virus": 0,
#     "Canine_distemper":    1,
#     "Mammary_Tumor":       3,
#     "Mange":               4,
# }
manual_index_map = None  # set to a dict like above if needed

# ---------------- Build file list ----------------
# Sanity: keep only classes that actually exist in data_root (optional)
available = {d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))}
class_names = [c for c in include_classes if c in available] or include_classes
num_classes = len(class_names)
print(f"Using classes (n={num_classes}): {class_names}")

def latent_path_for(class_name, new_idx):
    if manual_index_map is not None:
        old_idx = manual_index_map[class_name]
    else:
        # default: assume 0..3 correspond to the order in class_names
        old_idx = new_idx
    path = os.path.join(latent_dir, f"latent_class_{old_idx}.npy")
    return path, old_idx

# ------------- Load latents -------------
X_list, y_list = [], []
index_map_used = {}
for i, cname in enumerate(class_names):
    path, old_idx = latent_path_for(cname, i)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing latent file for {cname}: {path}")
    z = np.load(path)  # shape: (N_i, D)
    X_list.append(z)
    y_list.append(np.full(z.shape[0], i, dtype=int))
    index_map_used[cname] = old_idx

print("Index map used (class_name -> latent_class_<idx>.npy):")
for k, v in index_map_used.items():
    print(f"  {k} -> latent_class_{v}.npy")

X = np.vstack(X_list)            # (N_total, D)
y = np.concatenate(y_list)       # (N_total,)
print(f"Loaded latents: X={X.shape}, per-class counts={np.bincount(y)}")

# --------- Standardize then UMAP ----------
X_scaled = StandardScaler().fit_transform(X)

reducer = umap.UMAP(
    n_neighbors=15,      # try 10–50 if you want to explore
    min_dist=0.1,        # smaller = tighter clusters
    n_components=2,
    metric='euclidean',
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)  # (N, 2)

# --------- Save coordinates (CSV) ---------
df = pd.DataFrame({
    "umap_1": X_umap[:, 0],
    "umap_2": X_umap[:, 1],
    "label_id": y,
    "label_name": [class_names[i] for i in y]
})
df.to_csv("umap_latent_coordinates_4class.csv", index=False)
print("Saved: umap_latent_coordinates_4class.csv")

# --------------- Plot ----------------
plt.figure(figsize=(9, 7))
palette = sns.color_palette("Set2", n_colors=num_classes)

for i, name in enumerate(class_names):
    mask = (y == i)
    plt.scatter(
        X_umap[mask, 0], X_umap[mask, 1],
        s=8, alpha=0.75, label=name, color=palette[i]
    )
    if np.any(mask):
        cx, cy = X_umap[mask, 0].mean(), X_umap[mask, 1].mean()
        plt.text(cx, cy, name, fontsize=9, weight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.6))

plt.title("UMAP of SAE Latent Features (4 Diseases)", pad=10)
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.legend(markerscale=2, frameon=True)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("umap_latent_4class.png", dpi=300)
plt.show()

print("Saved: umap_latent_4class.png")
