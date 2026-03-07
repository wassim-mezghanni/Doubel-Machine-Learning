import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import statsmodels.api as sm

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# --- Refined Data Loading & Preprocessing ---

def preprocess_user_features(user_data):
    """
    Extracts and encodes features based on Comment-Robert schema.
    """
    # Mapping for Sunshine Credit
    credit_map = {
        "信用极好": 5,
        "信用较好": 4,
        "信用中等": 3,
        "信用一般": 2,
        "信用较差": 1,
        "信用极差": 0
    }
    
    features = []
    
    # Numerical
    features.append(float(user_data.get('followers_count', 0)))
    features.append(float(user_data.get('friends_count', 0)))
    features.append(float(user_data.get('statuses_count', 0)))
    features.append(float(user_data.get('mbrank', 0)))
    features.append(float(user_data.get('mbtype', 0)))
    
    # Sunshine Credit 
    credit_val = user_data.get('sunshine_credit', '信用一般')
    features.append(float(credit_map.get(credit_val, 2)))
    
    # Verified (Boolean)
    features.append(1.0 if user_data.get('verified', False) else 0.0)
    
    # Gender (Binary)
    gender = user_data.get('gender', 'f')
    features.append(1.0 if gender == 'm' else 0.0)
    
    # Location (Simple encoding - count of chars )
    loc = user_data.get('location', '其他')
    features.append(float(len(loc))) 
    
    # Label Description (Count of labels)
    labels = user_data.get('label_desc', [])
    features.append(float(len(labels)))
    
    return np.array(features)

def load_and_align_data(mapping_path, posts_path, users_path, embeddings_path):
    print("Loading data mapping...")
    id_mapping = pd.read_csv(mapping_path).head(50000)
    post_ids = id_mapping['_id'].values
    
    print("Loading embeddings...")
    embeddings = np.load(embeddings_path)
    
    print("Loading Posts.json...")
    with open(posts_path, 'r', encoding='utf-8') as f:
        posts_data = json.load(f)
    
    post_map = {p['_id']: {
        'likes_count': p.get('likes_count', 0),
        'user_id': p.get('user', {}).get('_id')
    } for p in posts_data}
    
    print("Loading Users.json...")
    with open(users_path, 'r', encoding='utf-8') as f:
        users_raw = json.load(f)
    user_map = {u['_id']: u for u in users_raw}
    
    aligned_likes = []
    aligned_user_features = []
    
    print("Aligning 50,000 samples...")
    for pid in post_ids:
        p_info = post_map.get(pid, {'likes_count': 0, 'user_id': None})
        aligned_likes.append(p_info['likes_count'])
        
        u_info = user_map.get(p_info['user_id'], {})
        u_feat = preprocess_user_features(u_info)
        aligned_user_features.append(u_feat)
        
    return embeddings, np.array(aligned_user_features), np.array(aligned_likes), post_ids

# --- Variational Autoencoder (VAE) ---

class VAE(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, latent_dim=100):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_vae(data, latent_dim=100, epochs=20, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=data.shape[1], latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training VAE for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x,) in enumerate(loader):
            x = x.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x)
            loss = vae_loss_function(recon_batch, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        all_mu, _ = model.encode(torch.FloatTensor(data).to(device))
    return all_mu.cpu().numpy()

# --- Double Machine Learning (DML) Branch Models ---

class BranchModel(nn.Module):
    def __init__(self, text_dim=100, user_dim=10):
        super(BranchModel, self).__init__()
        # Branch A: Text (MLP)
        self.branch_a = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 16) # Increased to 16D features for merging
        )
        # Branch B: User (Linear layer with bias)
        self.branch_b = nn.Linear(user_dim, 16, bias=True)

        # Final merging layer (FC layer on concatenated outputs)
        self.merger = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, text_x, user_x):
        feat_a = self.branch_a(text_x)
        feat_b = self.branch_b(user_x)
        combined = torch.cat([feat_a, feat_b], dim=1)
        return self.merger(combined)

def run_dml_analysis():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    MAPPING_PATH = os.path.join(BASE_DIR, "id_mapping_part_0.csv")
    POSTS_PATH = os.path.join(BASE_DIR, "Datasets", "Posts.json")
    USERS_PATH = os.path.join(BASE_DIR, "Datasets", "Users.json")
    EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings_part_0.npy")
    
    # 1. Load Data
    embeddings_raw, user_features, likes_raw, post_ids = load_and_align_data(
        MAPPING_PATH, POSTS_PATH, USERS_PATH, EMBEDDINGS_PATH
    )
    
    # Apply Log Transformation to target
    likes = np.log1p(likes_raw)
    
    # 2. VAE Training
    latents = train_vae(embeddings_raw, latent_dim=100, epochs=15)
    
    # 3. Cross-Fitting (3-Fold)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(likes))
    
    print("Starting 3-fold cross-fitting with refinements...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(latents)):
        print(f"--- Fold {fold+1} ---")
        
        # Prepare Data
        X_text_train = torch.FloatTensor(latents[train_idx])
        X_user_train = torch.FloatTensor(user_features[train_idx])
        y_train = torch.FloatTensor(likes[train_idx]).view(-1, 1)
        
        X_text_val = torch.FloatTensor(latents[val_idx])
        X_user_val = torch.FloatTensor(user_features[val_idx])
        
        # Initialize Model
        model = BranchModel(text_dim=100, user_dim=user_features.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training
        dataset = TensorDataset(X_text_train, X_user_train, y_train)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        model.train()
        for epoch in range(15): # Increased epochs for better convergence
            for b_text, b_user, b_y in loader:
                optimizer.zero_grad()
                pred = model(b_text, b_user)
                loss = criterion(pred, b_y)
                loss.backward()
                optimizer.step()
        
        # Prediction
        model.eval()
        with torch.no_grad():
            fold_pred = model(X_text_val, X_user_val).numpy().flatten()
            oof_predictions[val_idx] = fold_pred

    # 4. Metrics (on log scale)
    mse_log = mean_squared_error(likes, oof_predictions)
    mae_log = mean_absolute_error(likes, oof_predictions)
    r2_log = r2_score(likes, oof_predictions)
    
    # Metrics (on original scale)
    preds_orig = np.expm1(oof_predictions)
    likes_orig = likes_raw
    mse_orig = mean_squared_error(likes_orig, preds_orig)
    mae_orig = mean_absolute_error(likes_orig, preds_orig)
    r2_orig = r2_score(likes_orig, preds_orig)
    
    print(f"\nFinal Metrics (Log Scale):")
    print(f"MSE: {mse_log:.4f}")
    print(f"MAE: {mae_log:.4f}")
    print(f"R2:  {r2_log:.4f}")
    
    print(f"\nFinal Metrics (Original Scale):")
    print(f"MSE: {mse_orig:.2f}")
    print(f"MAE: {mae_orig:.2f}")
    print(f"R2:  {r2_orig:.4f}")
    
    results = {
        "metrics_log": {"mse": mse_log, "mae": mae_log, "r2": r2_log},
        "metrics_original": {"mse": mse_orig, "mae": mae_orig, "r2": r2_orig},
        "predictions": [
            {
                "post_id": pid, 
                "actual": float(act), 
                "predicted": float(pre),
                "actual_log": float(np.log1p(act)),
                "predicted_log": float(lp)
            }
            for pid, act, pre, lp in zip(post_ids, likes_orig, preds_orig, oof_predictions)
        ]
    }
    
    with open(os.path.join(BASE_DIR, "dml_results.json"), "w") as f:
        json.dump(results, f)
    print("\nResults saved to dml_results.json")

if __name__ == "__main__":
    run_dml_analysis()
