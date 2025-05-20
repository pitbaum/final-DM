import pandas as pd
import torch
from tqdm import tqdm
import json
import torch.nn as nn
import torch.nn.functional as F

# Load data
test_df = pd.read_csv("data/test.csv").drop(columns=["timestamp"])

with open("data/questions.json", 'r') as f:
    questions = json.load(f)

with open("data/concept.json", 'r') as f:
    concepts = json.load(f)

question_embeddings = torch.load("question_embeddings.pt")
concept_embeddings = torch.load("concept_embeddings.pt")

# Load training data to get user ID mapping
train_data = pd.read_csv("data/train.csv")
train_df = train_data.drop(columns=["timestamp"])
all_users = set(train_df["uid"])
user_id_to_index = {uid: idx for idx, uid in enumerate(all_users)}

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, user_emb_dim, question_emb_dim, concept_emb_dim, hidden_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(user_emb_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.content_mlp = nn.Sequential(
            nn.Linear(question_emb_dim + concept_emb_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, user_ids, question_embeddings, concept_embeddings):
        u_emb = self.user_embedding(user_ids)
        user_vec = self.user_mlp(u_emb)

        content_vec = self.content_mlp(torch.cat([question_embeddings, concept_embeddings], dim=1))

        combined = torch.cat([user_vec, content_vec], dim=1)
        return self.output_layer(combined).squeeze(1)

    def forward(self, user_ids, question_embeddings, concept_embeddings):
        u_emb = self.user_embedding(user_ids)  # (B, user_emb_dim)
        user_vec = self.user_mlp(u_emb)        # (B, hidden_dim // 2)

        content_input = torch.cat([question_embeddings, concept_embeddings], dim=1)
        content_vec = self.content_mlp(content_input)  # (B, hidden_dim // 2)

        # Combine user and content representations
        combined = torch.cat([user_vec, content_vec], dim=1)  # (B, hidden_dim)

        logits = self.output_layer(combined)
        return logits.squeeze(1)  # for BCEWithLogitsLoss

# Initialize model
question_dim = len(question_embeddings[0])
concept_dim = len(concept_embeddings[0])
num_users = len(user_id_to_index)
model = TwoTowerModel(num_users=num_users, user_emb_dim=256,
                          question_emb_dim=question_dim,
                          concept_emb_dim=concept_dim,
                          hidden_dim=512)
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Prediction loop
results = []
for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
    user_id = row["uid"]
    question_id = row["question_id"]
    concept_ids_str = row["concept_id"]

    if user_id not in user_id_to_index:
        continue  # skip unknown users

    user_index = torch.tensor([user_id_to_index[user_id]], dtype=torch.long)
    question_embedding = question_embeddings[question_id].unsqueeze(0)

    if isinstance(concept_ids_str, str) and concept_ids_str.strip():
        try:
            concept_ids = [int(cid) for cid in concept_ids_str.split("_") if cid.isdigit()]
            if concept_ids:
                concept_tensors = [concept_embeddings[cid] for cid in concept_ids]
                concept_embedding = torch.mean(torch.stack(concept_tensors), dim=0)
            else:
                concept_embedding = torch.zeros(concept_dim)
        except Exception:
            concept_embedding = torch.zeros(concept_dim)
    else:
        concept_embedding = torch.zeros(concept_dim)

    concept_embedding = concept_embedding.unsqueeze(0)

    with torch.no_grad():
        output = model(user_index, question_embedding, concept_embedding)
        probability = torch.sigmoid(output).item()

    results.append((user_id, probability))

# Save predictions
results_df = pd.DataFrame(results, columns=["uid", "response"])
results_df.to_csv("output.csv", index=False)
print("Predictions saved to output.csv")
