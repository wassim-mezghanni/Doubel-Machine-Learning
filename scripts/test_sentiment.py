import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import transformers
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

tests = ['great product I love it', 'terrible very bad', 'normal day', 'hello world', 'amazing beautiful']
for t in tests:
    inputs = tokenizer(t, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print(f"Text: {t}")
    print(f"  Logits: {logits.tolist()}")
    print(f"  Probs: {probs.tolist()}")
    print(f"  id2label: {model.config.id2label}")
    print(f"  num_labels: {model.config.num_labels}")
    print()

# Also test with a batch
batch_texts = ['I love this', 'I hate this', 'ok fine']
batch_inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
with torch.no_grad():
    batch_out = model(**batch_inputs)
batch_probs = torch.nn.functional.softmax(batch_out.logits, dim=-1)
print("Batch logits:", batch_out.logits.tolist())
print("Batch probs:", batch_probs.tolist())
