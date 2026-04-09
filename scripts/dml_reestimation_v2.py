"""
DML Re-estimation v2: Dual-Specification Analysis
====================================================
Specification 1: RobReplied as the baseline treatment
Specification 2: female as the baseline treatment

4 Dependent Variables: ln_likes, ln_comments, ln_others_comments, ln_commenters
5-fold cross-fitting, no intercept.
"""
import os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import statsmodels.api as sm

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

ROBOT_ID = "79f6c7ffc7270a3a7d3136245ab0f8ac"

# --- Data Loading ---

def preprocess_user_features(user_data):
    credit_map = {"信用极好": 5, "信用较好": 4, "信用中等": 3, "信用一般": 2, "信用较差": 1, "信用极差": 0}
    return [
        float(user_data.get('followers_count', 0)),
        float(user_data.get('friends_count', 0)),
        float(user_data.get('mbrank', 0)),
        float(credit_map.get(user_data.get('sunshine_credit', '信用一般'), 2)),
        float(len(user_data.get('label_desc', []))),
        1.0  # Constant bias feature
    ]

def load_all_data(base_dir):
    DATA_DIR = os.path.join(base_dir, "Datasets")
    mapping_path = os.path.join(DATA_DIR, "id_mapping_final.csv")
    posts_path = os.path.join(DATA_DIR, "Posts.json")
    users_path = os.path.join(DATA_DIR, "Users.json")
    comments_path = os.path.join(DATA_DIR, "Comments.json")
    clean_metrics_path = os.path.join(base_dir, "clean_comment_metrics.csv")

    print_flush("Loading id mapping...")
    id_mapping = pd.read_csv(mapping_path)
    post_ids = set(id_mapping['_id'].values)
    mapping_order = id_mapping['_id'].values

    print_flush("Loading Posts.json...")
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

    print_flush("Loading Users.json...")
    with open(users_path, 'r', encoding='utf-8') as f:
        users_raw = json.load(f)
    user_map = {u['_id']: u for u in users_raw}

    print_flush("Searching for Robot Replies in Comments.json...")
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
            print_flush("Memory Error... falling back.")

    # Load clean comment metrics
    print_flush("Loading clean_comment_metrics.csv...")
    clean_df = pd.read_csv(clean_metrics_path)
    clean_map = {row['_id']: row for _, row in clean_df.iterrows()}

    # Align everything
    aligned = {k: [] for k in [
        'user_features', 'rob_replied', 'gender', 'verified',
        'likes', 'comments', 'others_comments', 'commenters', 'ip_locations'
    ]}

    print_flush(f"Aligning {len(mapping_order)} samples...")
    for pid in mapping_order:
        p_info = post_map.get(pid, {})
        u_info = user_map.get(p_info.get('user_id'), {})

        aligned['user_features'].append(preprocess_user_features(u_info))
        is_rob = 1.0 if pid in rob_replied_ids else 0.0
        aligned['rob_replied'].append(is_rob)
        aligned['gender'].append(1.0 if u_info.get('gender', 'f') == 'f' else 0.0)
        aligned['verified'].append(1.0 if u_info.get('verified', False) else 0.0)
        aligned['likes'].append(p_info.get('likes_count', 0))

        c_count = p_info.get('comments_count', 0)
        if is_rob > 0.5:
            c_count = max(0, c_count - 1)
        aligned['comments'].append(c_count)

        cm = clean_map.get(pid, {})
        aligned['others_comments'].append(cm.get('clean_comments', 0) if isinstance(cm, dict) else 0)
        aligned['commenters'].append(cm.get('unique_commenters', 0) if isinstance(cm, dict) else 0)

        ip = str(p_info.get('ip_location', 'Unknown')).strip()
        aligned['ip_locations'].append(ip if ip else 'Unknown')

    # Time features
    print_flush("Processing time features...")
    p_times = [post_map.get(pid, {}).get('created_at', '2020-01-01 00:00:00') for pid in mapping_order]
    u_times = [user_map.get(post_map.get(pid, {}).get('user_id'), {}).get('created_at', '2020-01-01 00:00:00') for pid in mapping_order]

    p_dates = pd.to_datetime(p_times, errors='coerce').fillna(pd.Timestamp('2020-01-01'))
    u_dates = pd.to_datetime(u_times, errors='coerce').fillna(pd.Timestamp('2020-01-01'))
    account_ages = (p_dates - u_dates).total_seconds().to_numpy() / 86400.0
    min_date = p_dates.min()
    time_vals = (p_dates - min_date).total_seconds().to_numpy() / 86400.0

    ip_dummies = pd.get_dummies(aligned['ip_locations'], prefix='ip', dummy_na=False).values.astype(float)
    u_feat_matrix = np.array(aligned['user_features'])
    final_features = np.hstack([u_feat_matrix, account_ages.reshape(-1, 1), ip_dummies])

    return {
        "final_features": final_features,
        "likes": np.array(aligned['likes']),
        "comments": np.array(aligned['comments']),
        "others_comments": np.array(aligned['others_comments']),
        "commenters": np.array(aligned['commenters']),
        "rob_replied": np.array(aligned['rob_replied']),
        "gender": np.array(aligned['gender']),
        "verified": np.array(aligned['verified']),
        "time": time_vals,
        "p_dates": p_dates.values
    }

# --- Neural Network ---

class BranchModel(nn.Module):
    def __init__(self, text_dim=150, user_dim=10, output_type='regression'):
        super().__init__()
        self.output_type = output_type
        self.branch_a = nn.Sequential(nn.Linear(text_dim, 256), nn.ReLU(), nn.Linear(256, 16))
        self.branch_b = nn.Linear(user_dim, 16, bias=True)
        self.merger = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, tx, ux):
        combined = torch.cat([self.branch_a(tx), self.branch_b(ux)], dim=1)
        out = self.merger(combined)
        return torch.sigmoid(out) if self.output_type == 'classification' else out

def train_and_get_residuals(latents, u_feat, y_data, task_type, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    residuals = np.zeros(len(y_data))
    for fold, (train_idx, val_idx) in enumerate(kf.split(latents)):
        print_flush(f"      Fold {fold+1}/{n_folds}...")
        X_t_train = torch.FloatTensor(latents[train_idx])
        X_u_train = torch.FloatTensor(u_feat[train_idx])
        y_train = torch.FloatTensor(y_data[train_idx]).view(-1, 1)
        X_t_val = torch.FloatTensor(latents[val_idx])
        X_u_val = torch.FloatTensor(u_feat[val_idx])

        model = BranchModel(text_dim=latents.shape[1], user_dim=u_feat.shape[1], output_type=task_type)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss() if task_type == 'classification' else nn.MSELoss()

        loader = DataLoader(TensorDataset(X_t_train, X_u_train, y_train), batch_size=1024, shuffle=True)
        model.train()
        for epoch in range(5):
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

# --- OLS Helpers ---

def extract_ols(model):
    return {"coef": model.params.tolist(), "pvalues": model.pvalues.tolist(), "r2": model.rsquared}

DV_NAMES = ["ln_likes", "ln_comments", "ln_others_comments", "ln_commenters"]

def run_full_sample_regressions(res_dvs, res_base, res_fem_rob, verified, time_vals, spec_name):
    """Run models (1)-(4) for a single specification on all 4 DVs."""
    results = {}
    for dv_name, res_y in res_dvs.items():
        dv_results = {}

        # Model 1: res_Y = k1 * res_base
        X1 = res_base.reshape(-1, 1)
        dv_results["m1"] = extract_ols(sm.OLS(res_y, X1).fit())

        # Model 2: res_Y = k1 * res_base + k2 * res_female_rob
        X2 = np.column_stack([res_base, res_fem_rob])
        dv_results["m2"] = extract_ols(sm.OLS(res_y, X2).fit())

        # Model 3: verified interaction
        # res_Y = k11*res_base + k12*verified*res_base + k21*res_fem_rob + k22*verified*res_fem_rob
        X3 = np.column_stack([res_base, verified * res_base, res_fem_rob, verified * res_fem_rob])
        dv_results["m3"] = extract_ols(sm.OLS(res_y, X3).fit())

        # Model 4: time interaction
        X4 = np.column_stack([res_base, time_vals * res_base, res_fem_rob, time_vals * res_fem_rob])
        dv_results["m4"] = extract_ols(sm.OLS(res_y, X4).fit())

        results[dv_name] = dv_results
    return results

def run_subgroup_regressions(res_dvs, res_base, res_fem_rob):
    """Run model (5) for a single specification on all 4 DVs."""
    results = {}
    for dv_name, res_y in res_dvs.items():
        X5 = np.column_stack([res_base, res_fem_rob])
        results[dv_name] = {"m5": extract_ols(sm.OLS(res_y, X5).fit())}
    return results

# --- Main Pipeline ---

def run_dml_analysis():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"

    data = load_all_data(BASE_DIR)

    print_flush("Loading optimal VAE latents (dim 150)...")
    optimal_latents = np.load(os.path.join(BASE_DIR, "optimal_vae_latents.npy"))

    # Sort by post date
    sort_idx = np.argsort(data["p_dates"])

    latents    = optimal_latents[sort_idx]
    u_feat     = data["final_features"][sort_idx]
    ln_likes   = np.log1p(data["likes"][sort_idx])
    ln_comms   = np.log1p(data["comments"][sort_idx])
    ln_oc      = np.log1p(data["others_comments"][sort_idx])
    ln_ctr     = np.log1p(data["commenters"][sort_idx])
    rob        = data["rob_replied"][sort_idx]
    female     = data["gender"][sort_idx]
    verified   = data["verified"][sort_idx]
    time_vals  = data["time"][sort_idx]
    female_rob = female * rob

    n = len(sort_idx)
    idx16 = int(n * 0.16)
    idx50 = int(n * 0.50)

    groups = {
        "full_sample": (0, n),
        "group1_16pct": (0, idx16),
        "group2_34pct": (idx16, idx50),
        "group3_50pct": (idx50, n)
    }

    all_results = {"spec1_RobReplied": {}, "spec2_female": {}}

    for g_name, (start, end) in groups.items():
        print_flush(f"\n{'='*60}")
        print_flush(f"  Group: {g_name}  (n={end-start})")
        print_flush(f"{'='*60}")

        l_g = latents[start:end]
        u_g = u_feat[start:end]
        v_g = verified[start:end]
        t_g = time_vals[start:end]

        # --- Residualize 4 DVs ---
        print_flush("  [DV] Residualizing ln_likes...")
        r_likes = train_and_get_residuals(l_g, u_g, ln_likes[start:end], 'regression')
        print_flush("  [DV] Residualizing ln_comments...")
        r_comms = train_and_get_residuals(l_g, u_g, ln_comms[start:end], 'regression')
        print_flush("  [DV] Residualizing ln_others_comments...")
        r_oc    = train_and_get_residuals(l_g, u_g, ln_oc[start:end], 'regression')
        print_flush("  [DV] Residualizing ln_commenters...")
        r_ctr   = train_and_get_residuals(l_g, u_g, ln_ctr[start:end], 'regression')

        res_dvs = dict(zip(DV_NAMES, [r_likes, r_comms, r_oc, r_ctr]))

        # --- Residualize shared treatment: female_RobReplied ---
        print_flush("  [TX] Residualizing female_RobReplied...")
        r_fem_rob = train_and_get_residuals(l_g, u_g, female_rob[start:end], 'regression')

        # --- Residualize Spec 1 treatment: RobReplied ---
        print_flush("  [TX] Residualizing RobReplied...")
        r_rob = train_and_get_residuals(l_g, u_g, rob[start:end], 'classification')

        # --- Residualize Spec 2 treatment: female ---
        print_flush("  [TX] Residualizing female...")
        r_fem = train_and_get_residuals(l_g, u_g, female[start:end], 'classification')

        is_full = (g_name == "full_sample")

        # --- Spec 1: RobReplied baseline ---
        if is_full:
            print_flush("  [REG] Spec 1 (RobReplied) – Full Sample models 1-4...")
            all_results["spec1_RobReplied"][g_name] = run_full_sample_regressions(
                res_dvs, r_rob, r_fem_rob, v_g, t_g, "spec1")
        else:
            print_flush(f"  [REG] Spec 1 (RobReplied) – {g_name} model 5...")
            all_results["spec1_RobReplied"][g_name] = run_subgroup_regressions(
                res_dvs, r_rob, r_fem_rob)

        # --- Spec 2: female baseline ---
        if is_full:
            print_flush("  [REG] Spec 2 (female) – Full Sample models 1-4...")
            all_results["spec2_female"][g_name] = run_full_sample_regressions(
                res_dvs, r_fem, r_fem_rob, v_g, t_g, "spec2")
        else:
            print_flush(f"  [REG] Spec 2 (female) – {g_name} model 5...")
            all_results["spec2_female"][g_name] = run_subgroup_regressions(
                res_dvs, r_fem, r_fem_rob)

    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(BASE_DIR, f"dml_reestimation_results_{date_str}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print_flush(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_dml_analysis()
