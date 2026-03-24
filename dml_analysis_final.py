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
        float(len(user_data.get('label_desc', []))),
        1.0  # Constant bias feature for the neural network branch
    ]
    return feats

def load_and_align_full_data(mapping_path, posts_path, users_path, comments_path, embeddings_path):
    print_flush("Loading final data mapping...")
    id_mapping = pd.read_csv(mapping_path)
    post_ids = set(id_mapping['_id'].values)
    mapping_order = id_mapping['_id'].values
    
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

    aligned_likes, aligned_comments, aligned_user_features, aligned_rob_replied = [], [], [], []
    aligned_gender, aligned_verified, aligned_ip_locations = [], [], []
    
    print_flush(f"Aligning {len(mapping_order)} samples...")
    for pid in mapping_order:
        p_info = post_map.get(pid, {})
        u_info = user_map.get(p_info.get('user_id'), {})
        
        u_feat = preprocess_user_features(u_info)
        aligned_user_features.append(u_feat)
        
        is_rob = 1.0 if pid in rob_replied_ids else 0.0
        aligned_rob_replied.append(is_rob)
        
        gender_val = 1.0 if u_info.get('gender', 'f') == 'f' else 0.0
        aligned_gender.append(gender_val)
        
        verified_val = 1.0 if u_info.get('verified', False) else 0.0
        aligned_verified.append(verified_val)
        
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
    
    # Post time parameter (days since first post)
    min_date = p_dates.min()
    aligned_time = (p_dates - min_date).total_seconds().to_numpy() / 86400.0
    
    ip_dummies = pd.get_dummies(aligned_ip_locations, prefix='ip', dummy_na=False).values.astype(float)
    
    u_feat_matrix = np.array(aligned_user_features)
    age_matrix = aligned_account_ages.reshape(-1, 1)
    
    final_features = np.hstack([u_feat_matrix, age_matrix, ip_dummies])
    
    return {
        "final_features": final_features,
        "likes": np.array(aligned_likes),
        "comments": np.array(aligned_comments),
        "rob_replied": np.array(aligned_rob_replied),
        "gender": np.array(aligned_gender),
        "verified": np.array(aligned_verified),
        "time": np.array(aligned_time),
        "p_dates": p_dates.values
    }

class BranchModel(nn.Module):
    def __init__(self, text_dim=150, user_dim=10, output_type='regression'):
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
        
        model = BranchModel(text_dim=latents.shape[1], user_dim=u_feat.shape[1], output_type=task_type)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss() if task_type == 'classification' else nn.MSELoss()
        
        loader = DataLoader(TensorDataset(X_t_train, X_u_train, y_train), batch_size=256, shuffle=True)
        model.train()
        for epoch in range(15):
            for bt, bu, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bt, bu), by)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_t_val, X_u_val).numpy().flatten()
            residuals[val_idx] = y_data[val_idx] - preds
    return residuals

def extract_ols_res(model):
    return {"coef": model.params.tolist(), "pvalues": model.pvalues.tolist(), "r2": model.rsquared}

def run_dml_analysis():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    DATA_DIR = os.path.join(BASE_DIR, "Datasets")
    
    data_dict = load_and_align_full_data(
        os.path.join(DATA_DIR, "id_mapping_final.csv"),
        os.path.join(DATA_DIR, "Posts.json"),
        os.path.join(DATA_DIR, "Users.json"),
        os.path.join(DATA_DIR, "Comments.json"),
        os.path.join(DATA_DIR, "embeddings_final.npy")
    )
    
    print_flush("Loading optimal VAE latents (dim 150)...")
    optimal_latents = np.load(os.path.join(BASE_DIR, "optimal_vae_latents.npy"))
    
    # Data transformations
    p_dates = data_dict["p_dates"]
    sort_idx = np.argsort(p_dates)
    
    latents = optimal_latents[sort_idx]
    u_feat = data_dict["final_features"][sort_idx]
    
    ln_likes = np.log1p(data_dict["likes"][sort_idx])
    ln_comments = np.log1p(data_dict["comments"][sort_idx])
    
    rob = data_dict["rob_replied"][sort_idx]
    female = data_dict["gender"][sort_idx]
    verified = data_dict["verified"][sort_idx]
    time = data_dict["time"][sort_idx]
    
    # Interaction terms
    female_rob = female * rob
    female_ver = female * verified
    female_rob_ver = female_rob * verified
    female_time = female * time
    female_rob_time = female_rob * time
    
    n = len(sort_idx)
    idx16 = int(n * 0.16)
    idx50 = int(n * 0.50)
    
    groups = {
        "full_sample": (0, n),
        "group1": (0, idx16),
        "group2": (idx16, idx50),
        "group3": (idx50, n)
    }
    
    results = {}
    
    for g_name, (start, end) in groups.items():
        print_flush(f"\n--- Running Comprehensive Analysis for {g_name} ---")
        l_g, u_g = latents[start:end], u_feat[start:end]
        
        y_likes = ln_likes[start:end]
        y_comms = ln_comments[start:end]
        
        # Residual Generation
        print_flush("  Generating residuals...")
        res_likes = train_and_get_residuals(l_g, u_g, y_likes, 'regression')
        res_comms = train_and_get_residuals(l_g, u_g, y_comms, 'regression')
        
        res_fem = train_and_get_residuals(l_g, u_g, female[start:end], 'classification')
        res_fem_rob = train_and_get_residuals(l_g, u_g, female_rob[start:end], 'regression')
        
        if g_name == "full_sample":
            res_fem_ver = train_and_get_residuals(l_g, u_g, female_ver[start:end], 'regression')
            res_fem_rob_ver = train_and_get_residuals(l_g, u_g, female_rob_ver[start:end], 'regression')
            res_fem_time = train_and_get_residuals(l_g, u_g, female_time[start:end], 'regression')
            res_fem_rob_time = train_and_get_residuals(l_g, u_g, female_rob_time[start:end], 'regression')
            
            # Full sample specific estimations
            # (1) likes = k1*female
            m1_l = sm.OLS(res_likes, res_fem).fit()
            m1_l_b = sm.OLS(res_likes, sm.add_constant(res_fem)).fit()
            m1_c = sm.OLS(res_comms, res_fem).fit()
            m1_c_b = sm.OLS(res_comms, sm.add_constant(res_fem)).fit()
            
            # (2) likes = k1*female + k2*female_rob
            X2 = np.column_stack([res_fem, res_fem_rob])
            m2_l = sm.OLS(res_likes, X2).fit()
            m2_l_b = sm.OLS(res_likes, sm.add_constant(X2)).fit()
            m2_c = sm.OLS(res_comms, X2).fit()
            m2_c_b = sm.OLS(res_comms, sm.add_constant(X2)).fit()
            
            # (3) verified interaction
            X3 = np.column_stack([res_fem, res_fem_ver, res_fem_rob, res_fem_rob_ver])
            m3_l = sm.OLS(res_likes, X3).fit()
            m3_l_b = sm.OLS(res_likes, sm.add_constant(X3)).fit()
            m3_c = sm.OLS(res_comms, X3).fit()
            m3_c_b = sm.OLS(res_comms, sm.add_constant(X3)).fit()
            
            # (4) time interaction
            X4 = np.column_stack([res_fem, res_fem_time, res_fem_rob, res_fem_rob_time])
            m4_l = sm.OLS(res_likes, X4).fit()
            m4_l_b = sm.OLS(res_likes, sm.add_constant(X4)).fit()
            m4_c = sm.OLS(res_comms, X4).fit()
            m4_c_b = sm.OLS(res_comms, sm.add_constant(X4)).fit()
            
            results[g_name] = {
                "likes": {
                    "m1_no_b": extract_ols_res(m1_l), "m1_b": extract_ols_res(m1_l_b),
                    "m2_no_b": extract_ols_res(m2_l), "m2_b": extract_ols_res(m2_l_b),
                    "m3_no_b": extract_ols_res(m3_l), "m3_b": extract_ols_res(m3_l_b),
                    "m4_no_b": extract_ols_res(m4_l), "m4_b": extract_ols_res(m4_l_b)
                },
                "comments": {
                    "m1_no_b": extract_ols_res(m1_c), "m1_b": extract_ols_res(m1_c_b),
                    "m2_no_b": extract_ols_res(m2_c), "m2_b": extract_ols_res(m2_c_b),
                    "m3_no_b": extract_ols_res(m3_c), "m3_b": extract_ols_res(m3_c_b),
                    "m4_no_b": extract_ols_res(m4_c), "m4_b": extract_ols_res(m4_c_b)
                }
            }
        else:
            # Subgroup estimations
            # (5) likes = k1*female + k2*female_rob
            X5 = np.column_stack([res_fem, res_fem_rob])
            m5_l = sm.OLS(res_likes, X5).fit()
            m5_l_b = sm.OLS(res_likes, sm.add_constant(X5)).fit()
            m5_c = sm.OLS(res_comms, X5).fit()
            m5_c_b = sm.OLS(res_comms, sm.add_constant(X5)).fit()
            
            results[g_name] = {
                "likes": {"m5_no_b": extract_ols_res(m5_l), "m5_b": extract_ols_res(m5_l_b)},
                "comments": {"m5_no_b": extract_ols_res(m5_c), "m5_b": extract_ols_res(m5_c_b)}
            }
            
    with open(os.path.join(BASE_DIR, "dml_reestimation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print_flush("\nRe-estimation Results saved to dml_reestimation_results.json")

if __name__ == "__main__":
    run_dml_analysis()
