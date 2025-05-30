import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import os
import torch.nn as nn


"""
    ######################################
    ###########DATA PREPARATION###########
    ######################################
"""


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

"""
    Create question embeddings
"""

if not os.path.exists(os.getcwd()+"/question_embeddings.pt"):

    print("question embeddings dont exist yet, creating them")
    # Load the sentence transformer model
    
    model = SentenceTransformer('shibing624/text2vec-base-chinese') 
    model.eval()

    questions_list = []
    for i in tqdm(range(len(questions))):
        question = questions[str(i)]

        content = question.get("content", "")
        kc_routes = " > ".join(question.get("kc_routes", []))
        answer = "".join(question.get("answer", []))
        analysis = question.get("analysis", "")
        q_type = question.get("type", "")

        # Combine into a structured Chinese prompt for embedding
        full_text = (
            f"题目类型: {q_type}题；"
            f"知识点路径: {kc_routes}；"
            f"题目内容: {content}；"
            f"解析: {analysis}；"
            f"答案: {answer}"
        )

        questions_list.append(full_text)

    batch_size = 32
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(questions_list), batch_size), desc="Encoding questions"):
            combined_batch = questions_list[i:i+batch_size]

            # Encode the batch
            batch_embeddings = model.encode(
                combined_batch,
                batch_size=batch_size,
                convert_to_tensor=True,
            )
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all embeddings
    question_embeddings = torch.cat(embeddings, dim=0)
    print("Finished creating question embeddings")

    # Normalize question embeddings
    question_mean = question_embeddings.mean(dim=0)
    question_std = question_embeddings.std(dim=0) + 1e-6  # Avoid division by zero
    question_embeddings = (question_embeddings - question_mean) / question_std

    torch.save(question_embeddings, "question_embeddings.pt")
    print("Saved to file")

else:
    question_embeddings = torch.load("question_embeddings.pt")
    
"""
    Generate concept embeddings 
"""

if not os.path.exists(os.getcwd()+"/concept_embeddings.pt"):
    model = SentenceTransformer("shibing624/text2vec-base-chinese") # Using a more state of the art model for embedding
    model.eval()
    concept_list = []
    for i in tqdm(range(len(concepts))):
        concept = concepts[str(i)]
        concept_list.append(concept)

    batch_size = 32
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(concept_list), batch_size), desc="Encoding concepts"):
            combined_batch = concept_list[i:i+batch_size]

            # Encode the batch
            batch_embeddings = model.encode(
                combined_batch,
                batch_size=batch_size,
                convert_to_tensor=True,
            )
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all embeddings
    concept_embeddings = torch.cat(embeddings, dim=0)
    print("Finished creating question embeddings")

    # Normalize concept embeddings
    concept_mean = concept_embeddings.mean(dim=0)
    concept_std = concept_embeddings.std(dim=0) + 1e-6
    concept_embeddings = (concept_embeddings - concept_mean) / concept_std


    torch.save(concept_embeddings, "concept_embeddings.pt")
    print("Saved to file")

else:
    concept_embeddings = torch.load("concept_embeddings.pt")
    print("Read the concept embeddings from file")

print("All prepared files should exist now, exiting program")


"""
    ######################################
    ###########Training Model#############
    ######################################
"""


#Load the training data from the file
train_df = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

train_df["timestamp_norm"] = (train_df["timestamp"] - train_df["timestamp"].mean()) / test_data["timestamp"].std()
test_data["timestamp_norm"] = (test_data["timestamp"] - test_data["timestamp"].mean()) / test_data["timestamp"].std()

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

        # User tower
        self.user_mlp = nn.Sequential(
            nn.Linear(user_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Content tower
        self.content_mlp = nn.Sequential(
            nn.Linear(question_emb_dim + concept_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Combined = (user tower output) + (content tower output)
        combined_dim = hidden_dim  # Because each tower outputs hidden_dim // 2

        self.output_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),        
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_ids, question_embeddings, concept_embeddings, timestamp):
        u_emb = self.user_embedding(user_ids)                        # (batch_size, user_emb_dim)

        # User tower
        user_input = u_emb       # (batch_size, user_emb_dim + 16)
        user_vec = self.user_mlp(user_input)                        # (batch_size, hidden_dim // 2)

        # Content tower
        content_input = torch.cat([question_embeddings, concept_embeddings], dim=1)
        content_vec = self.content_mlp(content_input)               # (batch_size, hidden_dim // 2)

        combined = torch.cat([user_vec, content_vec], dim=1)        # (batch_size, hidden_dim)

        return self.output_layer(combined).squeeze(1)               # (batch_size,)

# Dimensions from embedding data
question_dim = len(question_embeddings[0])
concept_dim = len(concept_embeddings[0])

# Map user_id to index
user_id_to_index = {uid: idx for idx, uid in enumerate(all_users)}
num_users = len(user_id_to_index)

# Initialize model
model = TwoTowerModel(
    num_users=num_users,
    user_emb_dim=512+256,
    question_emb_dim=question_dim,
    concept_emb_dim=concept_dim,
    hidden_dim=512
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
model.train()
criterion = nn.BCEWithLogitsLoss(reduction='none')

alpha = 0.2  # exponential decay rate

BATCH_SIZE = 64
epochs = 10

for epoch in range(epochs):
    model.train()
    batch_concepts = []
    batch_questions = []
    batch_uid = []
    batch_labels = []
    batch_timestamp = []

    shuffled_df = train_df.sample(frac=1).reset_index(drop=True)
    epoch_loss = 0
    batch_count = 0

    for i, row in tqdm(shuffled_df.iterrows(), total=len(shuffled_df), desc=f"Training Epoch {epoch+1}"):

        user_id = row["uid"]
        question_id = row["question_id"]
        concept_ids_str = row["concept_id"]
        label = row["response"]
        timestamp = float(row["timestamp_norm"])

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
        batch_timestamp.append(torch.tensor(timestamp, dtype=torch.float))

        # --- When batch is full ---
        if len(batch_uid) == BATCH_SIZE:
            in_batch_concepts = torch.stack(batch_concepts)
            in_batch_questions = torch.stack(batch_questions)
            in_batch_uid = torch.stack(batch_uid)
            label_batch = torch.cat(batch_labels)
            in_batch_timestamp = torch.stack(batch_timestamp).unsqueeze(1)

            # Forward pass
            output = model(in_batch_uid, in_batch_questions, in_batch_concepts, in_batch_timestamp)

            # Per-sample loss (shape: [batch_size])
            losses = criterion(output, label_batch)

            # Calculate recency weights based on timestamp (exponential decay)
            timestamps = in_batch_timestamp.squeeze(1)  # (batch_size,)
            recency_weights = torch.exp(-alpha * (1 - timestamps))

            # Apply weights to losses
            weighted_losses = losses * recency_weights

            # Average weighted loss
            loss = weighted_losses.mean()

            # Backprop and optimize
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
            batch_timestamp.clear()

    # --- Process leftover batch ---
    if batch_uid:
        in_batch_concepts = torch.stack(batch_concepts)
        in_batch_questions = torch.stack(batch_questions)
        in_batch_uid = torch.stack(batch_uid)
        label_batch = torch.cat(batch_labels)
        in_batch_timestamp = torch.stack(batch_timestamp).unsqueeze(1)

        output = model(in_batch_uid, in_batch_questions, in_batch_concepts, in_batch_timestamp)

        losses = criterion(output, label_batch)
        timestamps = in_batch_timestamp.squeeze(1)
        recency_weights = torch.exp(-alpha * (1 - timestamps))
        weighted_losses = losses * recency_weights
        loss = weighted_losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "model_weights.pth")

"""
    ######################################
    ##########Create output###############
    ######################################
"""

# Load data
test_df = test_data

with open("data/questions.json", 'r') as f:
    questions = json.load(f)

with open("data/concept.json", 'r') as f:
    concepts = json.load(f)

question_embeddings = torch.load("question_embeddings.pt")
concept_embeddings = torch.load("concept_embeddings.pt")
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Prediction loop
results = []
for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
    user_id = row["uid"]
    question_id = row["question_id"]
    concept_ids_str = row["concept_id"]
    timestamp = torch.tensor([row["timestamp_norm"]],dtype=torch.float)

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
        output = model(user_index, question_embedding, concept_embedding, timestamp)
        probability = torch.sigmoid(output).item()

    results.append((user_id, probability))

# Save predictions
results_df = pd.DataFrame(results, columns=["uid", "response"])
results_df.to_csv("output.csv", index=False)
print("Predictions saved to output.csv")