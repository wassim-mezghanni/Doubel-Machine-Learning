"""
DML Sentiment Regressions
=========================
Integration of Erlangshen-Roberta sentiment scores into the DML framework.
"""
import os, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dml_reestimation_v2 import load_all_data, train_and_get_residuals

def print_flush(msg):
    print(msg)
    import sys
    sys.stdout.flush()

BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"

def extract_ols(model):
    return {"coef": model.params.tolist(), "pvalues": model.pvalues.tolist(), "r2": model.rsquared}

def run_sentiment_dml():
    # 1. Load basic aligned data
    data = load_all_data(BASE_DIR)
    
    # 2. Load optimal latents
    print_flush("Loading optimal VAE latents...")
    latents = np.load(os.path.join(BASE_DIR, "optimal_vae_latents.npy"))
    
    # 3. Time sort
    sort_idx = np.argsort(data["p_dates"])
    mapping_order = pd.read_csv(os.path.join(BASE_DIR, "Datasets", "id_mapping_final.csv"))['_id'].values
    valid_pids = set(mapping_order) # Note: load_all_data filters some, so mapping_order is already filtered inside load_all_data?
    # Actually load_all_data doesn't return the pids. Let's just use the parallel arrays.
    
    # Let's cleanly map sentiment scores to the parallel arrays
    print_flush("Loading sentiment_results.csv...")
    sent_df = pd.read_csv(os.path.join(BASE_DIR, "sentiment_results.csv"))
    # The order in load_all_data is based on valid_indices of mapping_order.
    # So we need to reconstruct the pid array for the aligned data.
    
    DATA_DIR = os.path.join(BASE_DIR, "Datasets")
    id_mapping = pd.read_csv(os.path.join(DATA_DIR, "id_mapping_final.csv"))
    original_pids = id_mapping['_id'].values
    with open(os.path.join(DATA_DIR, "Posts.json"), 'r', encoding='utf-8') as f:
        posts_data = json.load(f)
    valid_pids_set = set()
    for p in posts_data:
        content = p.get('content', '')
        if '//@评论罗伯特' in content:
            idx = content.find('//@评论罗伯特')
            if '@评论罗伯特' not in content[:idx]:
                continue
        valid_pids_set.add(p['_id'])
    
    aligned_pids = [pid for pid in original_pids if pid in valid_pids_set]
    aligned_pids = np.array(aligned_pids)[sort_idx] # sort by time
    
    # Map sentiment to sorted arrays
    sent_map = {row['_id']: row for _, row in sent_df.iterrows()}
    
    avg_clean_sent = []
    robot_sent = []
    
    for pid in aligned_pids:
        sm_row = sent_map.get(pid, {})
        avg_clean_sent.append(sm_row.get('avg_clean_sentiment', np.nan))
        robot_sent.append(sm_row.get('robot_comment_sentiment', np.nan))
        
    avg_clean_sent = np.array(avg_clean_sent, dtype=float)
    robot_sent = np.array(robot_sent, dtype=float)
    
    # Retrieve sorted arrays
    u_feat     = data["final_features"][sort_idx]
    l_feat     = latents[sort_idx]
    
    female     = data["gender"][sort_idx]
    rob        = data["rob_replied"][sort_idx]
    fem_rob    = female * rob
    
    ln_likes   = np.log1p(data["likes"][sort_idx])
    ln_comms   = np.log1p(data["comments"][sort_idx])
    ln_oc      = np.log1p(data["others_comments"][sort_idx])
    ln_ctr     = np.log1p(data["commenters"][sort_idx])
    
    results = {"phase1_avg_sentiment": {}, "phase2_sent_interaction": {}}
    
    # ==========================================
    # Phase 1: Average Sentiment DV Estimations
    # ==========================================
    print_flush("\n=== Phase 1: Average Sentiment DV ===")
    # Filter to posts with valid avg_clean_sentiment
    valid_idx = ~np.isnan(avg_clean_sent)
    print_flush(f"Found {valid_idx.sum()} posts with valid clean comments.")
    
    v_l_feat = l_feat[valid_idx]
    v_u_feat = u_feat[valid_idx]
    v_avg_sent = avg_clean_sent[valid_idx]
    v_fem = female[valid_idx]
    v_rob = rob[valid_idx]
    v_fem_rob = fem_rob[valid_idx]
    
    print_flush("[DV] Residualizing avg_clean_sentiment...")
    res_avg_sent = train_and_get_residuals(v_l_feat, v_u_feat, v_avg_sent, 'regression')
    print_flush("[TX] Residualizing female...")
    res_v_fem = train_and_get_residuals(v_l_feat, v_u_feat, v_fem, 'classification')
    print_flush("[TX] Residualizing RobReplied...")
    res_v_rob = train_and_get_residuals(v_l_feat, v_u_feat, v_rob, 'classification')
    print_flush("[TX] Residualizing female_RobReplied...")
    res_v_fem_rob = train_and_get_residuals(v_l_feat, v_u_feat, v_fem_rob, 'regression')
    
    print_flush("[REG] Running Phase 1 regressions...")
    # 1. avg = k1 * res_female
    m1 = sm.OLS(res_avg_sent, res_v_fem.reshape(-1, 1)).fit()
    results["phase1_avg_sentiment"]["m1_female"] = extract_ols(m1)
    
    # 2. avg = k1 * res_female + k2 * res_fem_rob
    m2 = sm.OLS(res_avg_sent, np.column_stack([res_v_fem, res_v_fem_rob])).fit()
    results["phase1_avg_sentiment"]["m2_female"] = extract_ols(m2)
    
    # 3. avg = k1 * res_RobReplied
    m3 = sm.OLS(res_avg_sent, res_v_rob.reshape(-1, 1)).fit()
    results["phase1_avg_sentiment"]["m3_rob"] = extract_ols(m3)
    
    # 4. avg = k1 * res_RobReplied + k2 * res_fem_rob
    m4 = sm.OLS(res_avg_sent, np.column_stack([res_v_rob, res_v_fem_rob])).fit()
    results["phase1_avg_sentiment"]["m4_rob"] = extract_ols(m4)
    
    # ==========================================
    # Phase 2: Engagement with Sentiment Interaction
    # ==========================================
    print_flush("\n=== Phase 2: Sentiment Interaction on Engagement ===")
    
    # Mean center robot_sentiment
    valid_rob_sent = robot_sent[~np.isnan(robot_sent)]
    mean_rob_sent = np.mean(valid_rob_sent) if len(valid_rob_sent) > 0 else 0
    print_flush(f"Mean robot sentiment: {mean_rob_sent:.4f}")
    
    rob_sent_centered = np.where(np.isnan(robot_sent), 0, robot_sent - mean_rob_sent)
    
    # Create interaction terms
    tx_rob_sent = rob * rob_sent_centered
    tx_fem_rob_sent = fem_rob * rob_sent_centered
    
    # Residualize 4 DVs
    print_flush("[DV] Residualizing ln_likes, ln_comments, ln_others, ln_commenters...")
    res_likes = train_and_get_residuals(l_feat, u_feat, ln_likes, 'regression')
    res_comms = train_and_get_residuals(l_feat, u_feat, ln_comms, 'regression')
    res_oc    = train_and_get_residuals(l_feat, u_feat, ln_oc, 'regression')
    res_ctr   = train_and_get_residuals(l_feat, u_feat, ln_ctr, 'regression')
    
    # Residualize primary treatments
    print_flush("[TX] Residualizing primary treatments...")
    res_rob = train_and_get_residuals(l_feat, u_feat, rob, 'classification')
    res_fem_rob = train_and_get_residuals(l_feat, u_feat, fem_rob, 'regression')
    
    # Residualize new interaction terms
    print_flush("[TX] Residualizing sentiment interaction terms...")
    res_tx_rob_sent = train_and_get_residuals(l_feat, u_feat, tx_rob_sent, 'regression')
    res_tx_fem_rob_sent = train_and_get_residuals(l_feat, u_feat, tx_fem_rob_sent, 'regression')
    
    dvs = {
        "ln_likes": res_likes,
        "ln_comments": res_comms,
        "ln_others_comments": res_oc,
        "ln_commenters": res_ctr
    }
    
    for name, res_y in dvs.items():
        # res_Y = k11*res_Rob + k12*res_Rob_Sent + k21*res_Fem_Rob + k22*res_Fem_Rob_Sent
        X = np.column_stack([res_rob, res_tx_rob_sent, res_fem_rob, res_tx_fem_rob_sent])
        model = sm.OLS(res_y, X).fit()
        results["phase2_sent_interaction"][name] = extract_ols(model)
        
    out_path = os.path.join(BASE_DIR, "dml_sentiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print_flush(f"\nSentiment DML results saved to {out_path}")

if __name__ == "__main__":
    run_sentiment_dml()
