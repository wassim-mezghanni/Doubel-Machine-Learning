import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

from dml_analysis_final import load_and_align_full_data, VAE, vae_loss_function

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def evaluate_vae(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (x,) in data_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss = vae_loss_function(recon, x, mu, logvar)
            total_loss += loss.item()
    return total_loss / len(data_loader.dataset)

def train_vae_grid(data, latent_dims=[25, 50, 75, 100, 125, 150, 175, 200], epochs=20):
    # Split data into train and test
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test)), batch_size=256, shuffle=False)
    
    oos_losses = {}
    best_loss = float('inf')
    best_dim = None
    best_model_state = None
    
    for l_dim in latent_dims:
        print_flush(f"\n--- Training VAE with latent dimension: {l_dim} ---")
        model = VAE(input_dim=data.shape[1], hidden_dim=512, latent_dim=l_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for (x,) in train_loader:
                x = x.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x)
                loss = vae_loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
        test_loss = evaluate_vae(model, test_loader, device)
        oos_losses[l_dim] = test_loss
        print_flush(f"  Out-of-Sample Loss for dim {l_dim}: {test_loss:.4f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_dim = l_dim
            best_model_state = model.state_dict()
            
    print_flush(f"\nOptimal Latent Dimension: {best_dim} with OOS Loss: {best_loss:.4f}")
    
    # Generate full latent representations using best model
    best_model = VAE(input_dim=data.shape[1], hidden_dim=512, latent_dim=best_dim).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    
    with torch.no_grad():
        mu, _ = best_model.encode(torch.FloatTensor(data).to(device))
    full_latents = mu.cpu().numpy()
    
    return oos_losses, best_dim, best_model, full_latents

def main():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    DATA_DIR = os.path.join(BASE_DIR, "Datasets")
    
    print_flush("Loading filtered full data...")
    emb, u_feat, likes, comments, rob_replied, gender, p_dates, ids = load_and_align_full_data(
        os.path.join(DATA_DIR, "id_mapping_final.csv"),
        os.path.join(DATA_DIR, "Posts.json"),
        os.path.join(DATA_DIR, "Users.json"),
        os.path.join(DATA_DIR, "Comments.json"),
        os.path.join(DATA_DIR, "embeddings_final.npy")
    )
    
    print_flush(f"Full filtered sample size: {len(emb)}")
    
    # Use dimensions between 25 and 150+
    dims = [25, 50, 75, 100, 125, 150, 175, 200]
    
    oos_losses, best_dim, best_model, full_latents = train_vae_grid(emb, latent_dims=dims, epochs=20)
    
    # Save latents
    latents_path = os.path.join(BASE_DIR, "optimal_vae_latents.npy")
    np.save(latents_path, full_latents)
    print_flush(f"Saved optimal latent representations (dim {best_dim}) to {latents_path}")
    
    # Save model
    model_path = os.path.join(BASE_DIR, "optimal_vae_model.pt")
    torch.save(best_model.state_dict(), model_path)
    print_flush(f"Saved optimal VAE model to {model_path}")
    
    # Save grid search results
    with open(os.path.join(BASE_DIR, "vae_grid_search_losses.json"), 'w') as f:
        json.dump(oos_losses, f, indent=4)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(dims, [oos_losses[d] for d in dims], marker='o', linestyle='-', color='b')
    plt.title("VAE Out-of-Sample Loss vs. Latent Dimension")
    plt.xlabel("Latent Space Dimension (from 1024 input)")
    plt.ylabel("Out-of-Sample Loss")
    
    # Add vertical line for optimal dimension
    plt.axvline(x=best_dim, color='r', linestyle='--', label=f'Optimal Dim = {best_dim}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(BASE_DIR, "vae_loss_vs_dimension.png")
    plt.savefig(plot_path)
    print_flush(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
