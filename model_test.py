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

class UserQuestionModel(nn.Module):
    def __init__(self, num_users, user_emb_dim=1024, question_emb_dim=768, concept_emb_dim=768, hidden_dim=1024):
        super(UserQuestionModel, self).__init__()

        # Learnable embedding for each user
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)

        # Total input dimension = user + question + concept
        input_dim = user_emb_dim + question_emb_dim + concept_emb_dim

        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(hidden_dim // 4, 1)  # Binary classification logit

    def forward(self, user_ids, question_embeddings, concept_embeddings):
        # Lookup user embedding
        u_emb = self.user_embedding(user_ids)  # (batch_size, user_emb_dim)

        # Concatenate all inputs
        x = torch.cat([u_emb, question_embeddings, concept_embeddings], dim=1)

        # Feed-forward through MLP
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        logits = self.fc4(x)
        return logits.squeeze(1)  # For BCEWithLogitsLoss

# Initialize model
question_dim = len(question_embeddings[0])
concept_dim = len(concept_embeddings[0])
num_users = len(user_id_to_index)
model = UserQuestionModel(num_users=num_users, user_emb_dim=320,
                          question_emb_dim=question_dim,
                          concept_emb_dim=concept_dim,
                          hidden_dim=128)
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
