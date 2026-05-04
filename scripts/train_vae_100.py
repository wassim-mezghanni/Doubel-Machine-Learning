"""
Train VAE with latent_dim=100
==============================
Trains a VAE on the filtered embeddings and saves:
  - models/vae_latents_100.npy  (N x 100)
  - models/vae_model_100.pt
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class VAE(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, latent_dim=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)

def get_valid_indices(base_dir):
    """Get valid post indices matching the main pipeline filtering."""
    DATA_DIR = os.path.join(base_dir, "Datasets")
    id_mapping = pd.read_csv(os.path.join(DATA_DIR, "id_mapping_final.csv"))
    mapping_order = id_mapping['_id'].values

    print_flush("Loading Posts.json for repost filtering...")
    with open(os.path.join(DATA_DIR, "Posts.json"), 'r', encoding='utf-8') as f:
        posts_data = json.load(f)

    valid_pids = set()
    for p in posts_data:
        content = p.get('content', '')
        if '//@评论罗伯特' in content:
            idx = content.find('//@评论罗伯特')
            if '@评论罗伯特' not in content[:idx]:
                continue
        valid_pids.add(p['_id'])

    valid_indices = [i for i, pid in enumerate(mapping_order) if pid in valid_pids]
    print_flush(f"Valid indices: {len(valid_indices)} out of {len(mapping_order)}")
    return valid_indices

def main():
    set_seed(42)
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    DATA_DIR = os.path.join(BASE_DIR, "Datasets")
    LATENT_DIM = 100
    EPOCHS = 20
    BATCH_SIZE = 256

    # Get valid indices
    valid_indices = get_valid_indices(BASE_DIR)

    # Load embeddings
    print_flush("Loading embeddings_final.npy...")
    embeddings = np.load(os.path.join(DATA_DIR, "embeddings_final.npy"))
    print_flush(f"Full embeddings shape: {embeddings.shape}")

    filtered_emb = embeddings[valid_indices]
    print_flush(f"Filtered embeddings shape: {filtered_emb.shape}")
    del embeddings  # free memory

    device = torch.device("cpu")  # CPU to avoid CUDA OOM with large embeddings
    print_flush(f"Using device: {device}")

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(filtered_emb, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test)), batch_size=BATCH_SIZE, shuffle=False)

    model = VAE(input_dim=filtered_emb.shape[1], hidden_dim=512, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print_flush(f"\nTraining VAE with latent_dim={LATENT_DIM} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        # Test loss
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for (x,) in test_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar)
                test_loss += loss.item() * x.size(0)

        train_avg = epoch_loss / len(X_train)
        test_avg = test_loss / len(X_test)
        print_flush(f"  Epoch {epoch+1}/{EPOCHS} — Train: {train_avg:.6f}, Test: {test_avg:.6f}")

    # Extract latent representations for all filtered data
    print_flush("\nExtracting 100-dim latent representations...")
    model.eval()
    all_latents = []
    full_loader = DataLoader(TensorDataset(torch.FloatTensor(filtered_emb)), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (x,) in full_loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            all_latents.append(mu.cpu().numpy())
    latents = np.concatenate(all_latents, axis=0)
    print_flush(f"Latent representations shape: {latents.shape}")

    # Save
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    latents_path = os.path.join(models_dir, "vae_latents_100.npy")
    np.save(latents_path, latents)
    print_flush(f"Saved latents to {latents_path}")

    model_path = os.path.join(models_dir, "vae_model_100.pt")
    torch.save(model.state_dict(), model_path)
    print_flush(f"Saved model to {model_path}")

    print_flush(f"\nDone! Final test loss: {test_avg:.6f}")

if __name__ == "__main__":
    main()
