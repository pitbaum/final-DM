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

class UserQuestionModel(nn.Module):
    def __init__(self, num_users, user_emb_dim, question_emb_dim, concept_emb_dim, hidden_dim):
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


question_dim = len(question_embeddings[0])
concept_dim = len(concept_embeddings[0])
# Model init being the concept embedding dimension and the question embedding

# Map user_id to index
user_id_to_index = {uid: idx for idx, uid in enumerate(all_users)}
num_users = len(user_id_to_index)

# Re-initialize model with corrected num_users
model = UserQuestionModel(num_users=num_users, user_emb_dim=1024//4,
                          question_emb_dim=question_dim,
                          concept_emb_dim=concept_dim,
                          hidden_dim=1024)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
model.train()

BATCH_SIZE = 32
epochs = 3

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