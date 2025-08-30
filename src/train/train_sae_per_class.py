# === individual_supervised_autoencoders_224.py ===
# One supervised autoencoder per class, trained only on images of that class (224×224)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# === Dataset Loader for Individual Class ===
class SingleClassDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# === Model Architecture (true 224 encode & reconstruct) ===
# 224 -> 112 -> 56 -> 28 with 3 stride-2 convs
class SupervisedAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(SupervisedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, latent_dim)         # 64*28*28 = 50,176 -> 128
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 28 * 28),        # 128 -> 50,176
            nn.ReLU(),
            nn.Unflatten(1, (64, 28, 28)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   # 112 -> 224
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# === Prepare and Train One SAE Per Class ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # CHANGED: 224×224
    transforms.ToTensor(),           # keep pixel range in [0,1]; no normalization for reconstruction
])

root_dir = 'CanineNIBLD23'
# safer: only take subfolders as classes
classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

for class_idx, class_name in enumerate(classes):
    print(f"\nTraining SAE for Class {class_idx} ({class_name})...")
    class_dir = os.path.join(root_dir, class_name)
    image_paths = [
        os.path.join(class_dir, fname)
        for fname in os.listdir(class_dir)
        if fname.lower().endswith(('jpg', 'jpeg', 'png'))
    ]

    dataset = SingleClassDataset(image_paths, transform=transform)
    # CHANGED: default to 32 to avoid OOM at 224×224; increase if memory allows
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)

    model = SupervisedAutoencoder(latent_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(100):
        total_loss = 0.0
        for xb in dataloader:
            xb = xb.to(device)
            x_recon, _ = model(xb)
            loss = criterion(x_recon, xb)  # true 224×224 reconstruction

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset) if len(dataset) else 0.0
        print(f"Epoch {epoch+1:03d}/100 | Recon Loss: {avg_loss:.6f}")

    # Save latent features (still 128-D per image)
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for xb in dataloader:
            xb = xb.to(device)
            _, z = model(xb)
            latent_vectors.append(z.cpu().numpy())

    latents = np.vstack(latent_vectors) if latent_vectors else np.empty((0, 128), dtype=np.float32)
    np.save(f"AE_224_RESULTS/latent_class_{class_idx}.npy", latents)
    print(f"Saved latent features for class {class_name} as latent_class_{class_idx}.npy | shape={latents.shape}")

