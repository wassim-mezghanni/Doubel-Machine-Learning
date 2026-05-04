"""
Descriptive Statistics & Correlation Matrix
=============================================
Generates descriptive stats and correlations for all variables used in the DML analysis.
"""
import os, json, sys
import numpy as np
import pandas as pd

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

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

def extract_province(ip):
    if not ip or ip == 'Unknown':
        return None
    ip = ip.strip()
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

def main():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    DATA_DIR = os.path.join(BASE_DIR, "Datasets")
    OUT_DIR = os.path.join(BASE_DIR, "results", "robustness")
    os.makedirs(OUT_DIR, exist_ok=True)

    credit_map = {"信用极好": 5, "信用较好": 4, "信用中等": 3, "信用一般": 2, "信用较差": 1, "信用极差": 0}

    # Load data
    print_flush("Loading id mapping...")
    id_mapping = pd.read_csv(os.path.join(DATA_DIR, "id_mapping_final.csv"))
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
        if pid in set(mapping_order):
            post_map[pid] = p
            if p.get('mblogid'):
                mblogid_to_id[p['mblogid']] = pid

    valid_indices = [i for i, pid in enumerate(mapping_order) if pid in valid_pids]
    mapping_order = mapping_order[valid_indices]

    print_flush("Loading Users.json...")
    with open(os.path.join(DATA_DIR, "Users.json"), 'r', encoding='utf-8') as f:
        user_map = {u['_id']: u for u in json.load(f)}

    print_flush("Loading Comments.json...")
    rob_replied_ids = set()
    with open(os.path.join(DATA_DIR, "Comments.json"), 'r', encoding='utf-8') as f:
        try:
            for c in json.load(f):
                if c.get('comment_user', {}).get('_id') == ROBOT_ID:
                    mbid = c.get('root_post_mblogid')
                    if mbid in mblogid_to_id:
                        rob_replied_ids.add(mblogid_to_id[mbid])
        except MemoryError:
            pass

    print_flush("Loading clean_comment_metrics.csv...")
    clean_df = pd.read_csv(os.path.join(BASE_DIR, "results", "clean_comment_metrics.csv"))
    clean_map = dict(zip(clean_df['_id'], clean_df.to_dict('records')))

    print_flush("Loading sentiment_results.csv...")
    sent_df = pd.read_csv(os.path.join(BASE_DIR, "results", "sentiment_results.csv"))
    sent_map = dict(zip(sent_df['_id'], sent_df.to_dict('records')))

    # Build variable arrays
    print_flush(f"Building variable arrays for {len(mapping_order)} observations...")
    rows = []
    for pid in mapping_order:
        p = post_map.get(pid, {})
        uid = p.get('user', {}).get('_id')
        u = user_map.get(uid, {})
        is_rob = 1.0 if pid in rob_replied_ids else 0.0
        female = 1.0 if u.get('gender', 'f') == 'f' else 0.0
        verified = 1.0 if u.get('verified', False) else 0.0
        likes = p.get('likes_count', 0)
        c_count = p.get('comments_count', 0)
        if is_rob > 0.5:
            c_count = max(0, c_count - 1)

        cm = clean_map.get(pid, {})
        oc = cm.get('clean_comments', 0) if isinstance(cm, dict) else 0
        commenters = cm.get('unique_commenters', 0) if isinstance(cm, dict) else 0

        ip = str(p.get('ip_location', 'Unknown')).strip() or 'Unknown'
        prov = extract_province(ip)
        srb = SRB_INFO.get(prov) if prov else np.nan
        mainland = 1.0 if prov and prov in SRB_INFO else 0.0

        bday = parse_birthday(u.get('birthday', ''))
        post_dt = pd.to_datetime(p.get('created_at', ''), errors='coerce')
        if bday and not pd.isna(post_dt):
            age_y = (post_dt - bday).days / 365.25
            poster_age = age_y if 5 < age_y < 100 else np.nan
        else:
            poster_age = np.nan

        p_time = pd.to_datetime(p.get('created_at', ''), errors='coerce')
        u_time = pd.to_datetime(u.get('created_at', ''), errors='coerce')
        account_age = (p_time - u_time).total_seconds() / 86400.0 if not pd.isna(p_time) and not pd.isna(u_time) else np.nan

        sr = sent_map.get(pid, {})
        avg_sent = sr.get('avg_clean_sentiment', np.nan) if isinstance(sr, dict) else np.nan
        rob_sent = sr.get('robot_comment_sentiment', np.nan) if isinstance(sr, dict) else np.nan

        rows.append({
            'female': female, 'rob_replied': is_rob, 'female_RobReplied': female * is_rob,
            'verified': verified,
            'likes': likes, 'comments': c_count,
            'others_comments': oc, 'commenters': commenters,
            'SRB': srb, 'mainland': mainland, 'poster_age': poster_age,
            'followers_count': u.get('followers_count', 0),
            'friends_count': u.get('friends_count', 0),
            'mbrank': u.get('mbrank', 0),
            'sunshine_credit': credit_map.get(u.get('sunshine_credit', '信用一般'), 2),
            'label_desc_count': len(u.get('label_desc', [])),
            'account_age_days': account_age,
            'avg_clean_sentiment': avg_sent,
            'robot_comment_sentiment': rob_sent,
        })

    df = pd.DataFrame(rows)
    print_flush(f"DataFrame shape: {df.shape}")

    # Descriptive statistics
    print_flush("Computing descriptive statistics...")
    desc = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc['N_valid'] = df.count()
    desc = desc[['N_valid', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    desc_path = os.path.join(OUT_DIR, "descriptive_statistics.csv")
    desc.to_csv(desc_path)
    print_flush(f"Saved to {desc_path}")
    print_flush("\n" + desc.to_string())

    # Correlation matrix
    print_flush("\nComputing correlation matrix...")
    corr = df.corr()
    corr_path = os.path.join(OUT_DIR, "correlation_matrix.csv")
    corr.to_csv(corr_path)
    print_flush(f"Saved to {corr_path}")

    print_flush("\nDone!")

if __name__ == "__main__":
    main()
