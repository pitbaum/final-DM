import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import json

"""
    Load data
"""

#Load the training data from the file
train_data = pd.read_csv("data/train.csv")
# Drop useless columns from the data provided
train_df = train_data.drop(columns=["timestamp"])
del train_data #Clean memory

test_data = pd.read_csv("data/test.csv").drop(columns=["timestamp"])

with open("data/questions.json", 'r') as f:
    questions = json.load(f)

with open("data/concept.json", 'r') as f:
    concepts = json.load(f)

# Register a set of all users
all_users = set(train_df["uid"])

# Register a set of all concepts (keys of the 'concepts' dictionary)
all_concepts = list(concepts.keys())

question_embeddings = torch.load("question_embeddings.pt")

concept_embeddings = torch.load("concept_embeddings.pt")

"""
    Define models
"""

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


question_dim = len(question_embeddings[0])
concept_dim = len(concept_embeddings[0])
# Model init being the concept embedding dimension and the question embedding

# Map user_id to index
user_id_to_index = {uid: idx for idx, uid in enumerate(all_users)}
num_users = len(user_id_to_index)

model = TwoTowerModel(
    num_users=num_users,
    user_emb_dim=256,
    question_emb_dim=question_dim,
    concept_emb_dim=concept_dim,
    hidden_dim=512  # or 1024 depending on your needs
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
model.train()

BATCH_SIZE = 32
epochs = 10

for epoch in range(epochs):
    model.train()
    batch_concepts = []
    batch_questions = []
    batch_uid = []
    batch_labels = []

    epoch_loss = 0
    batch_count = 0
    for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Training Epoch {epoch+1}"):
        user_id = row["uid"]
        question_id = row["question_id"]
        concept_ids_str = row["concept_id"]  # e.g., "12_45"
        label = row["response"]

        # Parse and average concept embeddings
        if isinstance(concept_ids_str, str) and concept_ids_str.strip():
            try:
                concept_ids = [int(cid) for cid in concept_ids_str.split("_") if cid.isdigit()]
                if concept_ids:
                    this_concept = [concept_embeddings[cid] for cid in concept_ids]
                    concept_embedding = torch.mean(torch.stack(this_concept), dim=0)
                else:
                    concept_embedding = torch.zeros(concept_dim)
            except (KeyError, IndexError):
                concept_embedding = torch.zeros(concept_dim)
        else:
            concept_embedding = torch.zeros(concept_dim)

        question_embedding = question_embeddings[question_id]
        label_tensor = torch.tensor([label], dtype=torch.float)
        user_index = torch.tensor(user_id_to_index[user_id], dtype=torch.long)

        batch_concepts.append(concept_embedding)
        batch_questions.append(question_embedding)
        batch_uid.append(user_index)
        batch_labels.append(label_tensor)

        if len(batch_uid) == BATCH_SIZE:
            in_batch_concepts = torch.stack(batch_concepts)
            in_batch_questions = torch.stack(batch_questions)
            in_batch_uid = torch.stack(batch_uid)
            label_batch = torch.cat(batch_labels)

            output = model(in_batch_uid, in_batch_questions, in_batch_concepts)
            loss = criterion(output, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            batch_count += 1

            # Clear batch
            batch_concepts.clear()
            batch_questions.clear()
            batch_uid.clear()
            batch_labels.clear()

    # Process leftover batch
    if batch_uid:
        in_batch_concepts = torch.stack(batch_concepts)
        in_batch_questions = torch.stack(batch_questions)
        in_batch_uid = torch.stack(batch_uid)
        label_batch = torch.cat(batch_labels)

        output = model(in_batch_uid, in_batch_questions, in_batch_concepts)
        loss = criterion(output, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "model_weights.pth")