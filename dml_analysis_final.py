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
import statsmodels.api as sm
import sys

# Ensure unbuffered output for real-time monitoring
def print_flush(msg):
    print(msg)
    sys.stdout.flush()

set_seed = lambda seed=42: (np.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None, os.environ.update({'PYTHONHASHSEED': str(seed)}))
set_seed(42)

ROBOT_ID = "79f6c7ffc7270a3a7d3136245ab0f8ac"

def preprocess_user_features(user_data):
    credit_map = {"信用极好": 5, "信用较好": 4, "信用中等": 3, "信用一般": 2, "信用较差": 1, "信用极差": 0}
    feats = [
        float(user_data.get('followers_count', 0)),
        float(user_data.get('friends_count', 0)),
        float(user_data.get('mbrank', 0)),
        float(credit_map.get(user_data.get('sunshine_credit', '信用一般'), 2)),
        float(len(user_data.get('label_desc', [])))
    ]
    return feats

def load_and_align_full_data(mapping_path, posts_path, users_path, comments_path, embeddings_path):
    print_flush("Loading final data mapping...")
    id_mapping = pd.read_csv(mapping_path)
    post_ids = set(id_mapping['_id'].values)
    mapping_order = id_mapping['_id'].values
    
    print_flush(f"Loading final embeddings ({embeddings_path})...")
    embeddings = np.load(embeddings_path)
    
    print_flush(f"Loading Posts.json ({posts_path})...")
    with open(posts_path, 'r', encoding='utf-8') as f:
        posts_data = json.load(f)
    
    post_map = {}
    mblogid_to_id = {}
    valid_pids = set()
    for p in posts_data:
        pid = p['_id']
        content = p.get('content', '')
        if '//@评论罗伯特' in content:
            idx = content.find('//@评论罗伯特')
            if '@评论罗伯特' not in content[:idx]:
                continue
        valid_pids.add(pid)
        if pid in post_ids:
            post_map[pid] = {
                'likes_count': p.get('likes_count', 0),
                'comments_count': p.get('comments_count', 0),
                'created_at': p.get('created_at', ''),
                'ip_location': p.get('ip_location', 'Unknown'),
                'user_id': p.get('user', {}).get('_id')
            }
            if p.get('mblogid'):
                mblogid_to_id[p.get('mblogid')] = pid

    valid_indices = [i for i, pid in enumerate(mapping_order) if pid in valid_pids]
    mapping_order = mapping_order[valid_indices]
    embeddings = embeddings[valid_indices]

    print_flush(f"Loading Users.json ({users_path})...")
    with open(users_path, 'r', encoding='utf-8') as f:
        users_raw = json.load(f)
    user_map = {u['_id']: u for u in users_raw}
    
    print_flush(f"Inference: Searching for Robot Replies in {comments_path}...")
    rob_replied_ids = set()
    with open(comments_path, 'r', encoding='utf-8') as f:
        try:
            comments_data = json.load(f)
            for c in comments_data:
                if c.get('comment_user', {}).get('_id') == ROBOT_ID:
                    mbid = c.get('root_post_mblogid')
                    if mbid in mblogid_to_id:
                        rob_replied_ids.add(mblogid_to_id[mbid])
        except MemoryError:
            print_flush("Memory Error... falling back scanned mode.")
            pass

    aligned_likes, aligned_comments, aligned_user_features, aligned_rob_replied, aligned_gender = [], [], [], [], []
    aligned_ip_locations = []
    
    print_flush(f"Aligning {len(mapping_order)} samples...")
    for pid in mapping_order:
        p_info = post_map.get(pid, {})
        u_info = user_map.get(p_info.get('user_id'), {})
        
        # Base features
        u_feat = preprocess_user_features(u_info)
        aligned_user_features.append(u_feat)
        
        is_rob = 1.0 if pid in rob_replied_ids else 0.0
        aligned_rob_replied.append(is_rob)
        
        gender_val = 1.0 if u_info.get('gender', 'f') == 'f' else 0.0
        aligned_gender.append(gender_val)
        
        aligned_likes.append(p_info.get('likes_count', 0))
        
        c_count = p_info.get('comments_count', 0)
        if is_rob > 0.5:
            c_count = max(0, c_count - 1)
        aligned_comments.append(c_count)
        
        ip = str(p_info.get('ip_location', 'Unknown')).strip()
        aligned_ip_locations.append(ip if ip else 'Unknown')
        
    print_flush("Processing time and IP dummies...")
    p_times = [post_map.get(pid, {}).get('created_at', '2020-01-01 00:00:00') for pid in mapping_order]
    u_times = [user_map.get(post_map.get(pid, {}).get('user_id'), {}).get('created_at', '2020-01-01 00:00:00') for pid in mapping_order]
    
    p_dates = pd.to_datetime(p_times, errors='coerce').fillna(pd.Timestamp('2020-01-01 00:00:00'))
    u_dates = pd.to_datetime(u_times, errors='coerce').fillna(pd.Timestamp('2020-01-01 00:00:00'))
    aligned_account_ages = (p_dates - u_dates).total_seconds().to_numpy() / 86400.0
    
    ip_dummies = pd.get_dummies(aligned_ip_locations, prefix='ip', dummy_na=False).values.astype(float)
    
    u_feat_matrix = np.array(aligned_user_features)
    age_matrix = aligned_account_ages.reshape(-1, 1)
    
    final_features = np.hstack([u_feat_matrix, age_matrix, ip_dummies])
    
    return embeddings, final_features, np.array(aligned_likes), np.array(aligned_comments), np.array(aligned_rob_replied), np.array(aligned_gender), p_dates.values, mapping_order

class VAE(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, latent_dim=100):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

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

def train_vae(data, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=data.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.FloatTensor(data)), batch_size=256, shuffle=True)
    print_flush("  Training VAE...")
    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss_function(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader.dataset)
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(torch.FloatTensor(data).to(device))
    return mu.cpu().numpy(), final_loss

class BranchModel(nn.Module):
    def __init__(self, text_dim=100, user_dim=10, output_type='regression'):
        super(BranchModel, self).__init__()
        self.output_type = output_type
        self.branch_a = nn.Sequential(nn.Linear(text_dim, 256), nn.ReLU(), nn.Linear(256, 16))
        self.branch_b = nn.Linear(user_dim, 16, bias=True)
        self.merger = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, tx, ux):
        combined = torch.cat([self.branch_a(tx), self.branch_b(ux)], dim=1)
        out = self.merger(combined)
        return torch.sigmoid(out) if self.output_type == 'classification' else out

def train_and_get_residuals(latents, u_feat, y_data, task_type):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    residuals = np.zeros(len(y_data))
    for fold, (train_idx, val_idx) in enumerate(kf.split(latents)):
        print_flush(f"    Fold {fold+1}...")
        X_t_train, X_u_train = torch.FloatTensor(latents[train_idx]), torch.FloatTensor(u_feat[train_idx])
        y_train = torch.FloatTensor(y_data[train_idx]).view(-1, 1)
        X_t_val, X_u_val = torch.FloatTensor(latents[val_idx]), torch.FloatTensor(u_feat[val_idx])
        
        model = BranchModel(text_dim=100, user_dim=u_feat.shape[1], output_type=task_type)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss() if task_type == 'classification' else nn.MSELoss()
        
        loader = DataLoader(TensorDataset(X_t_train, X_u_train, y_train), batch_size=128, shuffle=True)
        model.train()
        for epoch in range(15):
            for bt, bu, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bt, bu), by)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            residuals[val_idx] = y_data[val_idx] - model(X_t_val, X_u_val).numpy().flatten()
    return residuals

def run_group_analysis(emb_g, u_feat_g, likes_g, comments_g, rob_replied_g, gender_g, group_name):
    print_flush(f"\n--- Running Comprehensive Analysis for {group_name} ---")
    ln_likes = np.log1p(likes_g)
    ln_comments = np.log1p(comments_g)
    female_rob = gender_g * rob_replied_g
    
    latents, vae_loss = train_vae(emb_g)
    print_flush(f"  VAE Loss ({group_name}): {vae_loss:.4f}")
    
    print_flush(f"  Training residuals for {group_name}...")
    res_likes = train_and_get_residuals(latents, u_feat_g, ln_likes, 'regression')
    res_comments = train_and_get_residuals(latents, u_feat_g, ln_comments, 'regression')
    res_rob = train_and_get_residuals(latents, u_feat_g, rob_replied_g, 'classification')
    res_female = train_and_get_residuals(latents, u_feat_g, gender_g, 'classification')
    res_female_rob = train_and_get_residuals(latents, u_feat_g, female_rob, 'regression')
    
    print_flush(f"  Estimating 6 Models for {group_name}...")
    # Regressions
    m1 = sm.OLS(res_likes, sm.add_constant(res_rob)).fit()
    m2 = sm.OLS(res_likes, sm.add_constant(np.column_stack([res_rob, res_female_rob]))).fit()
    m_add1 = sm.OLS(res_likes, sm.add_constant(res_female)).fit()
    m_add2 = sm.OLS(res_likes, sm.add_constant(np.column_stack([res_female, res_female_rob]))).fit()
    m_add3 = sm.OLS(res_comments, sm.add_constant(res_female)).fit()
    m_add4 = sm.OLS(res_comments, sm.add_constant(np.column_stack([res_female, res_female_rob]))).fit()
    
    return {
        "vae_loss": float(vae_loss),
        "model1": {"coef": m1.params.tolist(), "pvalues": m1.pvalues.tolist(), "r2": m1.rsquared},
        "model2": {"coef": m2.params.tolist(), "pvalues": m2.pvalues.tolist(), "r2": m2.rsquared},
        "add_model1": {"coef": m_add1.params.tolist(), "pvalues": m_add1.pvalues.tolist(), "r2": m_add1.rsquared},
        "add_model2": {"coef": m_add2.params.tolist(), "pvalues": m_add2.pvalues.tolist(), "r2": m_add2.rsquared},
        "add_model3": {"coef": m_add3.params.tolist(), "pvalues": m_add3.pvalues.tolist(), "r2": m_add3.rsquared},
        "add_model4": {"coef": m_add4.params.tolist(), "pvalues": m_add4.pvalues.tolist(), "r2": m_add4.rsquared}
    }

def run_dml_analysis():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    DATA_DIR = os.path.join(BASE_DIR, "Datasets")
    
    emb, u_feat, likes, comments, rob_replied, gender, p_dates, ids = load_and_align_full_data(
        os.path.join(DATA_DIR, "id_mapping_final.csv"),
        os.path.join(DATA_DIR, "Posts.json"),
        os.path.join(DATA_DIR, "Users.json"),
        os.path.join(DATA_DIR, "Comments.json"),
        os.path.join(DATA_DIR, "embeddings_final.npy")
    )
    
    # Sorting by post creation time for technology diffusion analysis
    sort_idx = np.argsort(p_dates)
    emb, u_feat, likes, comments = emb[sort_idx], u_feat[sort_idx], likes[sort_idx], comments[sort_idx]
    rob_replied, gender = rob_replied[sort_idx], gender[sort_idx]
    
    n = len(sort_idx)
    idx16 = int(n * 0.16)
    idx50 = int(n * 0.50)
    
    groups = {
        "group1": (0, idx16),
        "group2": (idx16, idx50),
        "group3": (idx50, n)
    }
    
    diffusion_results = {}
    for g_name, (start, end) in groups.items():
        res = run_group_analysis(
            emb[start:end], u_feat[start:end], likes[start:end], 
            comments[start:end], rob_replied[start:end], gender[start:end], g_name
        )
        diffusion_results[g_name] = res
        
    with open(os.path.join(BASE_DIR, "dml_diffusion_results.json"), "w") as f:
        json.dump(diffusion_results, f, indent=4)
    print_flush("\nUnified Technology Diffusion Results (6 models per group) saved to dml_diffusion_results.json")

if __name__ == "__main__":
    run_dml_analysis()
