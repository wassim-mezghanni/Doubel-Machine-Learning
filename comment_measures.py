import os
import json
import ijson
import pandas as pd
import numpy as np
import sys

def print_flush(msg):
    print(msg)
    sys.stdout.flush()

ROBOT_ID = "79f6c7ffc7270a3a7d3136245ab0f8ac"

def main():
    BASE_DIR = r"c:\Users\ge27tuv\Projects\Doubel-Machine-Learning"
    DATA_DIR = os.path.join(BASE_DIR, "Datasets")
    
    posts_path = os.path.join(DATA_DIR, "Posts.json")
    comments_path = os.path.join(DATA_DIR, "Comments.json")
    mapping_path = os.path.join(DATA_DIR, "id_mapping_final.csv")
    
    print_flush("Loading id mapping...")
    id_mapping = pd.read_csv(mapping_path)
    post_ids_set = set(id_mapping['_id'].values)
    
    # post_map: mblogid -> {'_id': pid, 'author_id': uid, 'orig_comments': count}
    post_map = {}
    
    print_flush(f"Parsing Posts.json to build post map...")
    with open(posts_path, 'r', encoding='utf-8') as f:
        posts_data = json.load(f)
        for p in posts_data:
            pid = p['_id']
            # We only care about posts in our final sample
            if pid in post_ids_set:
                # Also apply the retweet filter just in case
                content = p.get('content', '')
                if '//@评论罗伯特' in content:
                    idx = content.find('//@评论罗伯特')
                    if '@评论罗伯特' not in content[:idx]:
                        continue
                
                mblogid = p.get('mblogid')
                if mblogid:
                    post_map[mblogid] = {
                        '_id': pid,
                        'author_id': p.get('user', {}).get('_id'),
                        'orig_comments': p.get('comments_count', 0),
                        'clean_comments_count': 0,
                        'unique_commenters_set': set()
                    }
                    
    print_flush(f"Valid posts to process: {len(post_map)}")
    
    print_flush("Iteratively parsing Comments.json...")
    processed_comments = 0
    with open(comments_path, 'rb') as f:
        # ijson.items yields each item in the root array
        for comment in ijson.items(f, 'item'):
            processed_comments += 1
            if processed_comments % 100_000 == 0:
                print_flush(f"  Processed {processed_comments} comments...")
            
            mbid = comment.get('root_post_mblogid')
            if mbid in post_map:
                c_user = comment.get('comment_user', {})
                c_uid = c_user.get('_id') if isinstance(c_user, dict) else None
                
                # Check exclusions
                if c_uid and c_uid != ROBOT_ID and c_uid != post_map[mbid]['author_id']:
                    post_map[mbid]['clean_comments_count'] += 1
                    post_map[mbid]['unique_commenters_set'].add(c_uid)
                    
    print_flush(f"Finished parsing {processed_comments} comments.")
    
    # Build dataframe
    records = []
    for mbid, info in post_map.items():
        records.append({
            '_id': info['_id'],
            'mblogid': mbid,
            'orig_comments': info['orig_comments'],
            'clean_comments': info['clean_comments_count'],
            'unique_commenters': len(info['unique_commenters_set'])
        })
        
    df = pd.DataFrame(records)
    
    # Calculate descriptive stats
    df['diff_clean'] = df['orig_comments'] - df['clean_comments']
    df['diff_unique'] = df['orig_comments'] - df['unique_commenters']
    
    print_flush("\n=== Descriptive Statistics ===")
    print_flush(df[['orig_comments', 'clean_comments', 'unique_commenters', 'diff_clean', 'diff_unique']].describe().to_string())
    
    out_path = os.path.join(BASE_DIR, "clean_comment_metrics.csv")
    df.to_csv(out_path, index=False)
    print_flush(f"\nSaved metrics to {out_path}")

if __name__ == "__main__":
    main()
