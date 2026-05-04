"""
DML Robustness Check
====================
Re-estimates all models under two VAE/fold configurations:
  Case 1: VAE dim=150, 10 folds
  Case 2: VAE dim=100, 5 folds

Models (full sample): m1-m8
Models (subgroups): m_sub
Sentiment as DV: s1-s4
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

ROBOT_ID = "79f6c7ffc7270a3a7d3136245ab0f8ac"

SRB_INFO = {
    "北京": 108.3, "天津": 108.7, "河北": 108.8, "山西": 105.1,
    "内蒙古": 107.0, "辽宁": 105.8, "吉林": 107.4, "黑龙江": 106.2,
    "上海": 108.0, "江苏": 109.0, "浙江": 110.2, "安徽": 113.1,
    "福建": 118.7, "江西": 120.3, "山东": 112.0, "河南": 108.4,
    "湖北": 114.3, "湖南": 114.2, "广东": 115.5, "广西": 114.5,
    "海南": 122.4, "重庆": 108.0, "四川": 108.2, "贵州": 113.2,
    "云南": 107.5, "西藏": 105.4, "陕西": 108.3, "甘肃": 107.5,
    "青海": 106.8, "宁夏": 105.7, "新疆": 107.5,
}

# ── helpers ──────────────────────────────────────────────────────

def extract_province(ip_location):
    if not ip_location or ip_location == 'Unknown':
        return None
    ip = ip_location.strip()
    if ip == "四川":
        return "四川"
    if ip.startswith("发布于"):
        prov = ip.replace("发布于 ", "").replace("发布于", "").strip()
        return prov if prov in SRB_INFO else None
    return None

def parse_birthday(bday_str):
    if not bday_str or not bday_str.strip():
        return None
    parts = bday_str.strip().split()
    try:
        dt = pd.to_datetime(parts[0], errors='coerce')
        if pd.isna(dt) or dt.year < 1950 or dt.year > 2015:
            return None
        return dt
    except Exception:
        return None

def preprocess_user_features(user_data):
    credit_map = {"信用极好": 5, "信用较好": 4, "信用中等": 3, "信用一般": 2, "信用较差": 1, "信用极差": 0}
    return [
        float(user_data.get('followers_count', 0)),
        float(user_data.get('friends_count', 0)),
        float(user_data.get('mbrank', 0)),
        float(credit_map.get(user_data.get('sunshine_credit', '信用一般'), 2)),
        float(len(user_data.get('label_desc', []))),
        1.0
    ]

# ── data loading ─────────────────────────────────────────────────

def load_all_data(base_dir):
    DATA_DIR = os.path.join(base_dir, "Datasets")
    print_flush("Loading id mapping...")
    id_mapping = pd.read_csv(os.path.join(DATA_DIR, "id_mapping_final.csv"))
    post_ids = set(id_mapping['_id'].values)
    mapping_order = id_mapping['_id'].values

    print_flush("Loading Posts.json...")
    with open(os.path.join(DATA_DIR, "Posts.json"), 'r', encoding='utf-8') as f:
        posts_data = json.load(f)

    post_map, mblogid_to_id, valid_pids = {}, {}, set()
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
                mblogid_to_id[p['mblogid']] = pid

    valid_indices = [i for i, pid in enumerate(mapping_order) if pid in valid_pids]
    mapping_order = mapping_order[valid_indices]

    print_flush("Loading Users.json...")
    with open(os.path.join(DATA_DIR, "Users.json"), 'r', encoding='utf-8') as f:
        user_map = {u['_id']: u for u in json.load(f)}

    print_flush("Loading Comments.json for robot replies...")
    rob_replied_ids = set()
    with open(os.path.join(DATA_DIR, "Comments.json"), 'r', encoding='utf-8') as f:
        try:
            for c in json.load(f):
                if c.get('comment_user', {}).get('_id') == ROBOT_ID:
                    mbid = c.get('root_post_mblogid')
                    if mbid in mblogid_to_id:
                        rob_replied_ids.add(mblogid_to_id[mbid])
        except MemoryError:
            print_flush("MemoryError in comments.")

    print_flush("Loading clean_comment_metrics.csv...")
    clean_df = pd.read_csv(os.path.join(base_dir, "results", "clean_comment_metrics.csv"))
    clean_map = dict(zip(clean_df['_id'], clean_df.to_dict('records')))

    print_flush("Loading sentiment_results.csv...")
    sent_df = pd.read_csv(os.path.join(base_dir, "results", "sentiment_results.csv"))
    sent_map = dict(zip(sent_df['_id'], sent_df.to_dict('records')))

    # Align
    keys = ['user_features','rob_replied','gender','verified','likes','comments',
            'others_comments','commenters','ip_locations','srb','mainland',
            'poster_age','avg_clean_sentiment','robot_comment_sentiment']
    aligned = {k: [] for k in keys}

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

        ip = str(p_info.get('ip_location', 'Unknown')).strip() or 'Unknown'
        aligned['ip_locations'].append(ip)

        prov = extract_province(ip)
        srb_val = SRB_INFO.get(prov) if prov else None
        aligned['srb'].append(srb_val if srb_val is not None else np.nan)
        aligned['mainland'].append(1.0 if srb_val is not None else 0.0)

        bday = parse_birthday(u_info.get('birthday', ''))
        post_dt = pd.to_datetime(p_info.get('created_at', ''), errors='coerce')
        if bday and not pd.isna(post_dt):
            age_y = (post_dt - bday).days / 365.25
            aligned['poster_age'].append(age_y if 5 < age_y < 100 else np.nan)
        else:
            aligned['poster_age'].append(np.nan)

        sr = sent_map.get(pid, {})
        aligned['avg_clean_sentiment'].append(sr.get('avg_clean_sentiment', np.nan) if isinstance(sr, dict) else np.nan)
        aligned['robot_comment_sentiment'].append(sr.get('robot_comment_sentiment', np.nan) if isinstance(sr, dict) else np.nan)

    # Time features
    print_flush("Processing time features...")
    p_times = [post_map.get(pid, {}).get('created_at', '2020-01-01 00:00:00') for pid in mapping_order]
    u_times = [user_map.get(post_map.get(pid, {}).get('user_id'), {}).get('created_at', '2020-01-01 00:00:00') for pid in mapping_order]
    p_dates = pd.to_datetime(p_times, errors='coerce').fillna(pd.Timestamp('2020-01-01'))
    u_dates = pd.to_datetime(u_times, errors='coerce').fillna(pd.Timestamp('2020-01-01'))
    account_ages = (p_dates - u_dates).total_seconds().to_numpy() / 86400.0
    time_vals = (p_dates - p_dates.min()).total_seconds().to_numpy() / 86400.0

    ip_dummies = pd.get_dummies(aligned['ip_locations'], prefix='ip', dummy_na=False).values.astype(float)
    u_feat = np.array(aligned['user_features'])
    feat_std = np.hstack([u_feat, account_ages.reshape(-1, 1), ip_dummies])
    feat_noip = np.hstack([u_feat, account_ages.reshape(-1, 1)])

    # Centre robot sentiment
    rs = np.array(aligned['robot_comment_sentiment'], dtype=float)
    rs_mean = np.nanmean(rs)
    robot_sent_c = np.where(np.isnan(rs), 0.0, rs - rs_mean)

    return {
        "feat_std": feat_std, "feat_noip": feat_noip,
        "likes": np.array(aligned['likes']),
        "comments": np.array(aligned['comments']),
        "others_comments": np.array(aligned['others_comments']),
        "commenters": np.array(aligned['commenters']),
        "rob_replied": np.array(aligned['rob_replied']),
        "gender": np.array(aligned['gender']),
        "verified": np.array(aligned['verified']),
        "time": time_vals, "p_dates": p_dates.values,
        "srb": np.array(aligned['srb'], dtype=float),
        "mainland": np.array(aligned['mainland']),
        "poster_age": np.array(aligned['poster_age'], dtype=float),
        "avg_clean_sentiment": np.array(aligned['avg_clean_sentiment'], dtype=float),
        "robot_sent_c": robot_sent_c,
        "valid_indices": valid_indices,
    }

# ── neural network ───────────────────────────────────────────────

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
    for fold, (tr, va) in enumerate(kf.split(latents)):
        print_flush(f"      Fold {fold+1}/{n_folds}...")
        Xt, Xu = torch.FloatTensor(latents[tr]), torch.FloatTensor(u_feat[tr])
        yt = torch.FloatTensor(y_data[tr]).view(-1, 1)
        Xvt, Xvu = torch.FloatTensor(latents[va]), torch.FloatTensor(u_feat[va])

        model = BranchModel(text_dim=latents.shape[1], user_dim=u_feat.shape[1], output_type=task_type)
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.BCELoss() if task_type == 'classification' else nn.MSELoss()
        loader = DataLoader(TensorDataset(Xt, Xu, yt), batch_size=1024, shuffle=True)
        model.train()
        for _ in range(5):
            for bt, bu, by in loader:
                opt.zero_grad(); loss = crit(model(bt, bu), by); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            residuals[va] = y_data[va] - model(Xvt, Xvu).numpy().flatten()
    return residuals

# ── OLS ──────────────────────────────────────────────────────────

def ols_res(model):
    return {"coef": model.params.tolist(), "pvalues": model.pvalues.tolist(), "r2": model.rsquared}

DV_NAMES = ["ln_likes", "ln_comments", "ln_others_comments", "ln_commenters"]

def residualize_all(latents, u_feat, dvs_raw, rob, female, fem_rob, n_folds):
    """Residualize 4 DVs + 3 treatments. Returns dict of residuals."""
    res = {}
    for name, y in dvs_raw.items():
        print_flush(f"    [DV] {name}...")
        res[name] = train_and_get_residuals(latents, u_feat, y, 'regression', n_folds)
    print_flush("    [TX] RobReplied...")
    res['rob'] = train_and_get_residuals(latents, u_feat, rob, 'classification', n_folds)
    print_flush("    [TX] female...")
    res['fem'] = train_and_get_residuals(latents, u_feat, female, 'classification', n_folds)
    print_flush("    [TX] female_RobReplied...")
    res['fem_rob'] = train_and_get_residuals(latents, u_feat, fem_rob, 'regression', n_folds)
    return res

def run_full_models(res, verified, time_vals, robot_sent_c):
    """m1-m4, m8 for both specs across 4 DVs."""
    out = {"spec1_RobReplied": {}, "spec2_female": {}}
    for spec_name, base_key in [("spec1_RobReplied", "rob"), ("spec2_female", "fem")]:
        rb = res[base_key]
        rfr = res['fem_rob']
        for dv in DV_NAMES:
            ry = res[dv]
            d = {}
            # m1
            d["m1"] = ols_res(sm.OLS(ry, rb.reshape(-1,1)).fit())
            # m2
            d["m2"] = ols_res(sm.OLS(ry, np.column_stack([rb, rfr])).fit())
            # m3 verified
            d["m3"] = ols_res(sm.OLS(ry, np.column_stack([rb, verified*rb, rfr, verified*rfr])).fit())
            # m4 time
            d["m4"] = ols_res(sm.OLS(ry, np.column_stack([rb, time_vals*rb, rfr, time_vals*rfr])).fit())
            # m8 robot sentiment
            d["m8"] = ols_res(sm.OLS(ry, np.column_stack([rb, robot_sent_c*rb, rfr, robot_sent_c*rfr])).fit())
            out[spec_name][dv] = d
    return out

def run_subgroup(res):
    """m_sub for both specs across 4 DVs."""
    out = {"spec1_RobReplied": {}, "spec2_female": {}}
    for spec_name, base_key in [("spec1_RobReplied", "rob"), ("spec2_female", "fem")]:
        rb, rfr = res[base_key], res['fem_rob']
        for dv in DV_NAMES:
            out[spec_name][dv] = {"m_sub": ols_res(sm.OLS(res[dv], np.column_stack([rb, rfr])).fit())}
    return out

def run_interaction_model(res, z_vals, model_name):
    """3-way interaction for both specs across 4 DVs."""
    out = {"spec1_RobReplied": {}, "spec2_female": {}}
    for spec_name, base_key in [("spec1_RobReplied", "rob"), ("spec2_female", "fem")]:
        rb, rfr = res[base_key], res['fem_rob']
        for dv in DV_NAMES:
            ry = res[dv]
            X = np.column_stack([rb, z_vals*rb, rfr, z_vals*rfr])
            out[spec_name][dv] = {model_name: ols_res(sm.OLS(ry, X).fit())}
    return out

def run_sentiment_dv(res):
    """Sentiment as DV: s1-s4."""
    out = {}
    ry = res['avg_sent']
    out["s1_female"] = ols_res(sm.OLS(ry, res['fem'].reshape(-1,1)).fit())
    out["s2_female_femrob"] = ols_res(sm.OLS(ry, np.column_stack([res['fem'], res['fem_rob']])).fit())
    out["s3_rob"] = ols_res(sm.OLS(ry, res['rob'].reshape(-1,1)).fit())
    out["s4_rob_femrob"] = ols_res(sm.OLS(ry, np.column_stack([res['rob'], res['fem_rob']])).fit())
    return out

# ── main pipeline ────────────────────────────────────────────────

def run_one_case(case_name, latents, data, n_folds, out_dir):
    set_seed(42)
    sort_idx = np.argsort(data["p_dates"])
    N = len(sort_idx)

    # Sort all arrays
    lat = latents[sort_idx]
    fs = data["feat_std"][sort_idx]
    fni = data["feat_noip"][sort_idx]
    ln_l = np.log1p(data["likes"][sort_idx])
    ln_c = np.log1p(data["comments"][sort_idx])
    ln_oc = np.log1p(data["others_comments"][sort_idx])
    ln_ct = np.log1p(data["commenters"][sort_idx])
    rob = data["rob_replied"][sort_idx]
    fem = data["gender"][sort_idx]
    ver = data["verified"][sort_idx]
    tv = data["time"][sort_idx]
    fem_rob = fem * rob
    srb = data["srb"][sort_idx]
    mainland = data["mainland"][sort_idx]
    page = data["poster_age"][sort_idx]
    avg_sent = data["avg_clean_sentiment"][sort_idx]
    rsent = data["robot_sent_c"][sort_idx]

    dvs_raw = dict(zip(DV_NAMES, [ln_l, ln_c, ln_oc, ln_ct]))
    results = {}

    # ── Phase A: Full sample, standard features ──
    print_flush(f"\n{'='*60}\n  Phase A: Full sample standard ({case_name})\n{'='*60}")
    res_a = residualize_all(lat, fs, dvs_raw, rob, fem, fem_rob, n_folds)
    results["full_sample"] = run_full_models(res_a, ver, tv, rsent)

    # Subgroups
    i16, i50 = int(N*0.16), int(N*0.50)
    for gname, (s, e) in [("group1_16pct",(0,i16)),("group2_34pct",(i16,i50)),("group3_50pct",(i50,N))]:
        print_flush(f"\n  Phase A subgroup: {gname} (n={e-s})")
        dv_g = {k: v[s:e] for k, v in dvs_raw.items()}
        res_g = residualize_all(lat[s:e], fs[s:e], dv_g, rob[s:e], fem[s:e], fem_rob[s:e], n_folds)
        results[gname] = run_subgroup(res_g)

    # ── Phase B: SRB interaction (no IP, SRB-matched subset) ──
    srb_mask = ~np.isnan(srb)
    n_srb = srb_mask.sum()
    print_flush(f"\n{'='*60}\n  Phase B: SRB interaction (n={n_srb})\n{'='*60}")
    dvs_srb = {k: v[srb_mask] for k, v in dvs_raw.items()}
    res_b = residualize_all(lat[srb_mask], fni[srb_mask], dvs_srb,
                            rob[srb_mask], fem[srb_mask], fem_rob[srb_mask], n_folds)
    results["srb_interaction"] = run_interaction_model(res_b, srb[srb_mask], "m5")
    results["srb_interaction"]["sample_size"] = int(n_srb)

    # ── Phase C: Mainland interaction (no IP, all obs) ──
    print_flush(f"\n{'='*60}\n  Phase C: Mainland interaction (n={N})\n{'='*60}")
    res_c = residualize_all(lat, fni, dvs_raw, rob, fem, fem_rob, n_folds)
    results["mainland_interaction"] = run_interaction_model(res_c, mainland, "m6")

    # ── Phase D: Poster age (standard features, valid-age subset) ──
    age_mask = ~np.isnan(page)
    n_age = age_mask.sum()
    print_flush(f"\n{'='*60}\n  Phase D: Poster age interaction (n={n_age})\n{'='*60}")
    dvs_age = {k: v[age_mask] for k, v in dvs_raw.items()}
    res_d = residualize_all(lat[age_mask], fs[age_mask], dvs_age,
                            rob[age_mask], fem[age_mask], fem_rob[age_mask], n_folds)
    results["poster_age_interaction"] = run_interaction_model(res_d, page[age_mask], "m7")
    results["poster_age_interaction"]["sample_size"] = int(n_age)

    # ── Phase E: Sentiment as DV (standard features, valid avg_sent subset) ──
    sent_mask = ~np.isnan(avg_sent)
    n_sent = sent_mask.sum()
    print_flush(f"\n{'='*60}\n  Phase E: Sentiment as DV (n={n_sent})\n{'='*60}")
    print_flush("    [DV] avg_clean_sentiment...")
    r_asent = train_and_get_residuals(lat[sent_mask], fs[sent_mask], avg_sent[sent_mask], 'regression', n_folds)
    print_flush("    [TX] female...")
    r_sfem = train_and_get_residuals(lat[sent_mask], fs[sent_mask], fem[sent_mask], 'classification', n_folds)
    print_flush("    [TX] RobReplied...")
    r_srob = train_and_get_residuals(lat[sent_mask], fs[sent_mask], rob[sent_mask], 'classification', n_folds)
    print_flush("    [TX] female_RobReplied...")
    r_sfr = train_and_get_residuals(lat[sent_mask], fs[sent_mask], fem_rob[sent_mask], 'regression', n_folds)
    res_e = {'avg_sent': r_asent, 'fem': r_sfem, 'rob': r_srob, 'fem_rob': r_sfr}
    results["sentiment_as_dv"] = run_sentiment_dv(res_e)
    results["sentiment_as_dv"]["sample_size"] = int(n_sent)

    # Save
    out_path = os.path.join(out_dir, f"{case_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print_flush(f"\nResults saved to {out_path}")

def main():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    OUT_DIR = os.path.join(BASE_DIR, "results", "robustness")
    os.makedirs(OUT_DIR, exist_ok=True)

    print_flush("Loading all data...")
    data = load_all_data(BASE_DIR)

    print_flush("Loading VAE latents (150-dim)...")
    lat150 = np.load(os.path.join(BASE_DIR, "models", "optimal_vae_latents.npy"))
    print_flush(f"  150-dim latents shape: {lat150.shape}")

    print_flush("Loading VAE latents (100-dim)...")
    lat100 = np.load(os.path.join(BASE_DIR, "models", "vae_latents_100.npy"))
    print_flush(f"  100-dim latents shape: {lat100.shape}")

    # Case 1: dim=150, folds=10
    print_flush("\n" + "="*70)
    print_flush("  CASE 1: VAE dim=150, 10 folds")
    print_flush("="*70)
    run_one_case("case1_dim150_fold10", lat150, data, n_folds=10, out_dir=OUT_DIR)

    # Case 2: dim=100, folds=5
    print_flush("\n" + "="*70)
    print_flush("  CASE 2: VAE dim=100, 5 folds")
    print_flush("="*70)
    run_one_case("case2_dim100_fold5", lat100, data, n_folds=5, out_dir=OUT_DIR)

    print_flush("\n\nAll robustness checks complete!")

if __name__ == "__main__":
    main()
