import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import os
import torch.nn.functional as F

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
