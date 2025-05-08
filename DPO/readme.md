Here's a **training pipeline to classify "chosen" vs. "rejected" responses using DPO-style reward scoring**, built on top of a causal language model (e.g., GPT-2) with **binary cross-entropy loss**.

---

## 🧱 Training Pipeline Overview

* 🔍 **Goal**: Classify responses (given a prompt) as *preferred* (1) or *not preferred* (0).
* 📊 **Method**: Train a scoring model $\hat{r}_\theta(x, y) = \beta (\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x))$, and use BCE loss.
* ✅ **Output**: A model that scores responses, with higher scores meaning more likely to be “chosen.”

---

## ✅ Code: `dpo_classifier.py`

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from tqdm import tqdm

# --------------------
# Config
# --------------------
model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
beta = 0.1
lr = 5e-6
num_epochs = 3

# --------------------
# Load models and tokenizer
# --------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# --------------------
# Load and prepare data
# --------------------
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
dataset = dataset.filter(lambda x: x["chosen"] is not None and x["rejected"] is not None)

# Create classification-style dataset
def to_classification_pairs(example):
    return [
        {"prompt": example["prompt"], "response": example["chosen"], "label": 1},
        {"prompt": example["prompt"], "response": example["rejected"], "label": 0}
    ]

dataset = dataset.map(to_classification_pairs, batched=True).shuffle(seed=42)

# Dataloader
def collate_fn(batch):
    prompts = [b["prompt"] for b in batch]
    responses = [b["response"] for b in batch]
    labels = [b["label"] for b in batch]
    return prompts, responses, torch.tensor(labels, dtype=torch.float32)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# --------------------
# Utilities
# --------------------
def get_log_probs(model, tokenizer, prompts, responses):
    inputs = [p + r for p, r in zip(prompts, responses)]
    encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)
    labels = input_ids[:, 1:]
    log_probs = log_probs[:, :-1, :]

    token_log_probs = log_probs.gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:]

    log_prob_sums = (token_log_probs * mask).sum(dim=1)
    return log_prob_sums

# Classification loss
def classification_loss(pi_logp, ref_logp, labels, beta=0.1):
    logits = beta * (pi_logp - ref_logp)
    return F.binary_cross_entropy_with_logits(logits, labels.to(device)), logits

# --------------------
# Training loop
# --------------------
optimizer = AdamW(model.parameters(), lr=lr)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for prompts, responses, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        pi_logp = get_log_probs(model, tokenizer, prompts, responses)
        ref_logp = get_log_probs(ref_model, tokenizer, prompts, responses)

        loss, logits = classification_loss(pi_logp, ref_logp, labels, beta)

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels.to(device)).sum().item()
        total += len(labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    acc = correct / total
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Save model
model.save_pretrained("dpo_classifier_model")
tokenizer.save_pretrained("dpo_classifier_model")
```

---

## 🔍 Output & Usage

Once trained, you can:

* Feed in `(prompt, response)`
* Compute $\hat{r}_\theta(x, y)$
* Apply a sigmoid + threshold to predict `chosen` (1) or `rejected` (0)

---

## 🧠 Summary

| Component   | Role                           |
| ----------- | ------------------------------ |
| `model`     | Learns to score responses      |
| `ref_model` | Provides baseline log-probs    |
| `loss`      | BCE loss over DPO-style reward |
| `labels`    | 1 (chosen), 0 (rejected)       |
| `output`    | Binary classifier via scoring  |

