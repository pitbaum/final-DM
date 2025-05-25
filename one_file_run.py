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

        # User tower
        self.user_mlp = nn.Sequential(
            nn.Linear(user_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Content tower
        self.content_mlp = nn.Sequential(
            nn.Linear(question_emb_dim + concept_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Combined input dimension = (user tower output) + (content tower output) = hidden_dim // 2
        combined_dim = hidden_dim


        self.output_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),        
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),            # final output
        )

    def forward(self, user_ids, question_embeddings, concept_embeddings):
        u_emb = self.user_embedding(user_ids)
        user_vec = self.user_mlp(u_emb)  # (batch_size, hidden_dim // 4)

        content_input = torch.cat([question_embeddings, concept_embeddings], dim=1)
        content_vec = self.content_mlp(content_input)  # (batch_size, hidden_dim // 4)

        combined = torch.cat([user_vec, content_vec], dim=1)  # (batch_size, hidden_dim // 2)

        return self.output_layer(combined).squeeze(1)

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

BATCH_SIZE = 64
epochs = 20

for epoch in range(epochs):
    model.train()
    batch_concepts = []
    batch_questions = []
    batch_uid = []
    batch_labels = []

    # Shuffle the training data at the start of each epoch
    shuffled_df = train_df.sample(frac=1).reset_index(drop=True)
    epoch_loss = 0
    batch_count = 0
    for i, row in tqdm(shuffled_df.iterrows(), total=len(shuffled_df), desc=f"Training Epoch {epoch+1}"):
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


"""
    ######################################
    ##########Create output###############
    ######################################
"""

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