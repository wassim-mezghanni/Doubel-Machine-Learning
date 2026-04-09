import json
import pandas as pd
import torch
import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import transformers
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Configuration
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_sentiment_model():
    print(f"Loading model {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tokenizer, model

def get_sentiment_scores(texts, tokenizer, model):
    """
    Returns scores for a batch of texts.
    The model outputs 2 classes: negative (0) and positive (1).
    We'll return the probability of being positive.
    """
    if not texts:
        return []
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Probability of class 1 (Positive)
        scores = probs[:, 1].cpu().numpy().tolist()
    return scores

def run_sentiment_extraction():
    # 1. Load Data
    print("Loading Posts.json...")
    with open("Datasets/Posts.json", "r", encoding="utf-8") as f:
        posts_data = json.load(f)
    
    print("Loading Comments.json...")
    with open("Datasets/Comments.json", "r", encoding="utf-8") as f:
        comments_data = json.load(f)

    tokenizer, model = load_sentiment_model()

    # 2. Extract Unique Texts to minimize inference calls
    # Note: We need to map back to IDs, so it's easier to process items directly but in batches.
    
    # Process Posts
    print("Processing post sentiments...")
    post_sentiments = {}
    post_texts = [p.get("content", "") for p in posts_data]
    post_ids = [p.get("_id") for p in posts_data]
    
    for i in tqdm(range(0, len(post_texts), BATCH_SIZE)):
        batch_texts = post_texts[i:i+BATCH_SIZE]
        batch_ids = post_ids[i:i+BATCH_SIZE]
        scores = get_sentiment_scores(batch_texts, tokenizer, model)
        for pid, score in zip(batch_ids, scores):
            post_sentiments[pid] = score

    # Map mblogid to _id
    mblogid_to_id = {}
    for p in posts_data:
        if p.get('mblogid'):
            mblogid_to_id[p.get('mblogid')] = p['_id']
            
    print("Processing comment sentiments...")
    post_to_comment_scores = {pid: [] for pid in post_ids}
    post_to_robot_sentiment = {pid: None for pid in post_ids}
    
    flat_comments = []
    for c in comments_data:
        mbid = c.get('root_post_mblogid')
        pid = mblogid_to_id.get(mbid)
        if pid and pid in post_ids:
            flat_comments.append({
                "post_id": pid,
                "text": c.get("content", ""),
                "user": c.get("comment_user", {}).get("_id", ""),
                "is_robot": c.get("comment_user", {}).get("_id", "") == "79f6c7ffc7270a3a7d3136245ab0f8ac" or c.get("comment_user", {}).get("name", "") == "评论罗伯特"
            })

    
    # Original Poster mapping
    op_map = {p["_id"]: p.get("user", {}).get("_id") for p in posts_data}
    
    print(f"Total comments to process: {len(flat_comments)}")
    for i in tqdm(range(0, len(flat_comments), BATCH_SIZE)):
        batch = flat_comments[i:i+BATCH_SIZE]
        texts = [b["text"] for b in batch]
        scores = get_sentiment_scores(texts, tokenizer, model)
        
        for b, score in zip(batch, scores):
            pid = b["post_id"]
            if b["is_robot"]:
                # If multiple robot comments, take the first one found
                if post_to_robot_sentiment[pid] is None:
                    post_to_robot_sentiment[pid] = score
            elif b["user"] != op_map.get(pid):
                # Clean comment (not robot, not OP)
                post_to_comment_scores[pid].append(score)

    # 3. Aggregate Results
    results = []
    for pid in post_ids:
        clean_scores = post_to_comment_scores.get(pid, [])
        avg_clean = sum(clean_scores) / len(clean_scores) if clean_scores else None
        
        results.append({
            "_id": pid,
            "post_sentiment": post_sentiments.get(pid),
            "avg_clean_sentiment": avg_clean,
            "robot_comment_sentiment": post_to_robot_sentiment.get(pid)
        })
    
    df = pd.DataFrame(results)
    df.to_csv("sentiment_results.csv", index=False)
    print("Sentiment results saved to sentiment_results.csv")

if __name__ == "__main__":
    run_sentiment_extraction()
