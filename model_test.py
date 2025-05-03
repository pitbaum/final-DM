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

user_known_embeddings = torch.load("known_user_embeddings.pt")
user_unknown_embeddings = torch.load("unknown_user_embeddings.pt")

"""
    Define models
"""

class RecommenderModel(nn.Module):
    def __init__(self, input_dim):
        super(RecommenderModel, self).__init__()
        
        # First hidden layer (input size is 756)
        self.fc1 = nn.Linear(input_dim, 2048)  # Increased size for larger input
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.4)

        # Second hidden layer
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.4)

        # Third hidden layer
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.3)

        # Fourth hidden layer
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.3)

        # Fifth hidden layer
        self.fc5 = nn.Linear(256, 128)

        # Final output layer
        self.fc6 = nn.Linear(128, 1)  # For predicting binary interaction (click/no-click)

    def forward(self, x):
        # Pass through the first hidden layer
        x = F.gelu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        # Pass through the second hidden layer
        x = F.gelu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        # Pass through the third hidden layer
        x = F.gelu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        # Pass through the fourth hidden layer
        x = F.gelu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout4(x)

        # Pass through the fifth hidden layer
        x = F.gelu(self.fc5(x))

        # Output layer for binary classification
        x = self.fc6(x)
        
        return x  # Logit for BCEWithLogitsLoss

linear = True

if linear:
    """Test linear model"""

    user_known_embeddings_length = len(user_known_embeddings[15027])
    user_unknown_embeddings_length = len(user_unknown_embeddings[15027])
    model = RecommenderModel(input_dim=user_known_embeddings_length + user_unknown_embeddings_length + len(question_embeddings[0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()  

    results = []
    for _, behavior in tqdm(test_data.iterrows(), total=len(test_data), desc=f"Calculating predictions"):
        user_id = behavior["uid"]
        question_id = behavior["question_id"]
        predictions = []
        

        if user_id not in user_known_embeddings.keys():
            this_known_embedding = torch.zeros(user_known_embeddings_length, dtype=torch.float32)
        if user_id not in user_unknown_embeddings.keys():
            this_unknown_embedding = torch.zeros(user_unknown_embeddings_length, dtype=torch.float32)
        if len(question_embeddings) < question_id:
            continue
        this_known_embedding = user_known_embeddings[user_id]
        this_unknown_embedding = user_unknown_embeddings[user_id]
        this_question = question_embeddings[question_id]
        input_tensor = torch.cat((this_known_embedding, this_unknown_embedding,this_question), dim=0)
            
        output = model(input_tensor)
        predictions.append((user_id,torch.sigmoid(output).detach().cpu().item()))

        results.append(predictions)

    print("Results calculated for all users.")

columns = ["uid","response"]
results = pd.DataFrame(results, columns=columns)
results.to_csv("output.csv", index=False)
